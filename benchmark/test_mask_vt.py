import os
import sys

CURRENT_FILE = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_FILE))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import collections


from utils.utils import setup_seed, weight_init

setup_seed(seed=42)


from dataset.MVSADataset import MVSADataset_mask, MVSADataset_tmdc_mask, MVSA_t2dr_Dataset
from torch.utils.data import DataLoader

import argparse
from models.basic_model import AVClassifier_AUXI_TV
from sklearn.metrics import f1_score 
import numpy as np 

from models.tmdc import build_model_vt
from main_t2dr_noise_vt import AVClassifier_basic


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='KineticSound', type=str,
                        help='CREMAD, KineticSound, etc.')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    # 数据路径
    parser.add_argument('--audio_path', default='/root/autodl-tmp/data/CREMA-D/AudioWVT/', type=str)
    parser.add_argument('--visual_path', default='/root/autodl-tmp/data/CREMA-D/', type=str)
    parser.add_argument('--max_length', default=128, type=int)

    # 训练参数
    parser.add_argument('--batch_size', default=32, type=int)       # 16
    parser.add_argument('--epochs', default=100, type=int)          # 200
    parser.add_argument('--total_epoch', default=10, type=int)

    # 优化器参数
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    # 噪声参数
    parser.add_argument('--visual_missing_rate', default=0.1, type=float, help='missing rate for visual')
    parser.add_argument('--text_missing_rate', default=0.1, type=float, help='missing rate for audio')
    parser.add_argument('--modality_missing', default=0.1, type=float, help='missing rate for audio')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size for masking')
    parser.add_argument('--apply_roll', default=False, type=bool, help='apply window shift')
    parser.add_argument('--gamma', type=float, default=4.0)
    parser.add_argument('--beta', type=float, default=1e-5)
    parser.add_argument('--max', type=int, default=1e20)
    parser.add_argument('--drop', default=0, type=int)


    # 保存路径
    parser.add_argument('--ckpt_path', default='/root/autodl-tmp/my/OGM_PE_ARL/results/iadr_new/', type=str, help='path to svte trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    
    # 设备参数
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--device', default='cuda', type=str, help='device to use')

    # 其他参数
    parser.add_argument('--cylcle_epoch', default=80, type=int, help='cycle for learning rate')
    parser.add_argument('--num_classes', default=34, type=int, help='number of classes')
    parser.add_argument('--intra_missing', default='Mask', type=str, help='Mask or Gaussian')
    parser.add_argument('--modelname', default='T2DR', type=str, help='number of classes')

    # 模型参数
    parser.add_argument('--pretrain', default=False, type=bool, help='resnet pretrain')
    parser.add_argument('--intra_dim', default=256, type=int, help='intra_dim')
    parser.add_argument('--test_checkpoint', default='/root/autodl-tmp/my/OGM_PE_ARL/oge_results/best_model_of_dataset_CREMAD_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_98_acc_0.7400568181818182.pth', type=str, help='for benchmark')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--iedr_dim', default=512, type=int, help='iedr_dim')
    parser.add_argument('--a_noise_intensity_level', default='clean', type=str, help='a_noise_intensity_level')
    parser.add_argument('--v_noise_intensity_level', default='clean', type=str, help='v_noise_intensity_level')
    parser.add_argument('--modality_missing_prob', default=0.0, type=int, help='modality_missing_prob')
    parser.add_argument('--modality', type=str, default='full')

    parser.add_argument('--hidden', type=int, default=256, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=34, help='number of classes [defined by args.dataset]')
    parser.add_argument('--num_heads', type=int, default=2, help='')


    # HME模型相关
    parser.add_argument("--d_all", type=int, default=128)  # aaaaaa
    parser.add_argument("--ACOUSTIC_DIM", type=int, default=512)
    parser.add_argument("--VISUAL_DIM", type=int, default=512)
    parser.add_argument("--output_dim", type=int, default=6)
    # parser.add_argument("--num_heads", type=int, default=2)      # 2
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--latent_layers", type=int, default=2)
    parser.add_argument("--hyper_depth", type=int, default=2)   # aaaaaa
    parser.add_argument("--num_latents", type=int, default=4)     # 4  
    # parser.add_argument("--latent_dim", type=int, default=64)
    
    # 注意力相关
    parser.add_argument("--attn_dropout", type=float, default=0.5)
    parser.add_argument("--attn_dropout_v", type=float, default=0.2)
    parser.add_argument("--attn_dropout_a", type=float, default=0.2)
    parser.add_argument("--relu_dropout", type=float, default=0.3)
    parser.add_argument("--res_dropout", type=float, default=0.3)
    parser.add_argument("--embed_dropout", type=float, default=0.2)

    parser.add_argument('--bypass_attn', default=True, type=int, help='bypass_attn')
    parser.add_argument('--iedr_stage', default=1, type=int, help='patch size for masking')
    
    # HME特定参数
    parser.add_argument("--similarity_threshold", type=float, default=0.8)
    parser.add_argument("--uncertainty_LB", type=float, default=0.2)
    parser.add_argument("--uncertainty_UB", type=float, default=1.0)
    parser.add_argument("--weights_threshold", type=float, default=0.2)

     # RedCore模型相关
    parser.add_argument('--input_dim_a', type=int, default=512, 
                       help='音频输入特征维度 (CREMAD: 128)')
    parser.add_argument('--input_dim_v', type=int, default=512, 
                       help='视觉输入特征维度 (ResNet特征: 2048)')
    
    parser.add_argument('--embd_size_a', type=int, default=256, 
                       help='音频嵌入维度')
    parser.add_argument('--embd_size_v', type=int, default=256, 
                       help='视觉嵌入维度')
    
    parser.add_argument('--embd_method_a', type=str, default='maxpool', 
                       choices=['last', 'maxpool', 'attention'],
                       help='音频嵌入方法')
    parser.add_argument('--embd_method_v', type=str, default='maxpool', 
                       choices=['last', 'maxpool', 'attention'],
                       help='视觉嵌入方法')
    
    return parser.parse_args()



def test(args, model, device, dataloader):
    """验证模型"""
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data in dataloader:
            # import ipdb; ipdb.set_trace();
            if args.modelname == 'IGML':
                spec, images, labels, _, _ = data
                spec = {k: v.to(device) for k, v in spec.items()}
                images = images.to(device)
                labels = labels.to(device)

                # TODO: make it simpler and easier to extend
                a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, out_a, out_v,a_std_fc,v_std_fc = model(spec, images.float())


            elif args.modelname == 'T2DR':
                # CramedDataset_t2dr_mask
                # images, spec, labels, images_pixel_mask, spec_pixel_mask
                text_ids, images, labels, text_mask, images_pixel_mask = data
                # import ipdb; ipdb.set_trace();
                text_ids = text_ids.to(device)
                images = images.to(device)
                labels = labels.to(device)
                text_mask = text_mask.to(device)
                images_pixel_mask = images_pixel_mask.to(device)

                out = model(
                    text_ids,
                    images,
                    text_mask,
                    images_pixel_mask,
                    labels=labels,
                    training=False
                )

                    
            elif args.modelname == 'TMDC':
                
                text_ids = data[0]['input_ids']
                text_mask = data[0]['attention_mask']
                visual = data[1]
                labels = data[2]
                vt_avail = data[3]

                batch_size = visual.shape[0]

                # import ipdb; ipdb.set_trace();
                vt_avail = torch.stack(vt_avail)
                vt_avail = torch.transpose(vt_avail, 0, 1)
                input_features_mask = vt_avail.unsqueeze(1)

                visual = visual.float().to(args.device)
                input_features_mask = input_features_mask.float().to(args.device)
                labels = labels.long().to(args.device)
                text_ids = text_ids.long().to(args.device)
                text_mask = text_mask.long().to(args.device)
                
                # ========== forward ==========
                hidden, out, outputs_intras, outputs_inters, vib_outputs, vib_kls = model(
                    visual,
                    text_ids,
                    text_mask,
                    input_features_mask,
                    first_stage=False
                )
                if out.dim() == 3 and out.shape[1] == 1:
                    out = out.squeeze(1)

            # import ipdb; ipdb.set_trace();
            preds = torch.argmax(out, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    # import ipdb; ipdb.set_trace();
    total_acc = correct / total if total > 0 else 0.0
    
    f1 = 0.0
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    f1 = f1_score(all_labels_np, all_preds_np, average='weighted')


    return total_acc, f1



def main():
    args = get_arguments()
    print("Arguments:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # mvsa-single
    weights_IGML_mvsa = "/root/autodl-tmp/data/results/mvsa/best_model_of_dataset_MVSA_Single_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_78_acc_0.7364864864864864.pth"
    test_scenarios_mvsa = {
        "IGML_v0.0_t0.0_vt": (0.0, 0.0, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.0_t0.0_t": (0.0, 0.0, 't', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.0_t0.0_v": (0.0, 0.0, 'v', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.1_t0.1_vt": (0.1, 0.1, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.1_t0.1_t": (0.1, 0.1, 't', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.1_t0.1_v": (0.1, 0.1, 'v', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.5_t0.5_vt": (0.5, 0.5, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.5_t0.5_t": (0.5, 0.5, 't', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.5_t0.5_v": (0.5, 0.5, 'v', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.1_t0.5_vt": (0.1, 0.5, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.5_t0.1_vt": (0.5, 0.1, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.1_t0.3_vt": (0.1, 0.3, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.3_t0.1_vt": (0.3, 0.1, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.3_t0.7_vt": (0.3, 0.7, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
        "IGML_v0.7_t0.3_vt": (0.7, 0.3, 'vt', "IGML", weights_IGML_mvsa, 'Mask'),
    }

    
    weights_TMDC_mvsa = "/root/autodl-tmp/data/results/mvsa_tmdc/MVSA_Single_best_second_stage_acc0.7324_epochs100_epoch76.pth"
    test_tmdc_scenarios_mvsa = {
        "TMDC_v0.0_t0.0_vt": (0.0, 0.0, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.0_t0.0_t": (0.0, 0.0, 't', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.0_t0.0_v": (0.0, 0.0, 'v', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.1_t0.1_vt": (0.1, 0.1, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.1_t0.1_t": (0.1, 0.1, 't', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.1_t0.1_v": (0.1, 0.1, 'v', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.5_t0.5_vt": (0.5, 0.5, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.5_t0.5_t": (0.5, 0.5, 't', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.5_t0.5_v": (0.5, 0.5, 'v', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.1_t0.5_vt": (0.1, 0.5, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.5_t0.1_vt": (0.5, 0.1, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.1_t0.3_vt": (0.1, 0.3, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.3_t0.1_vt": (0.3, 0.1, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.3_t0.7_vt": (0.3, 0.7, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
        "TMDC_v0.7_t0.3_vt": (0.7, 0.3, 'vt', "TMDC", weights_TMDC_mvsa, 'Mask'),
    }

    weights_T2DR_mvsa = "/root/autodl-tmp/data/results/mvsa_t2dr_fanhua/final_best_model_MVSA_Single_epoch88_acc0.7268.pth"
    test_t2dr_scenarios_mvsa = {
        "T2DR_v0.0_t0.0_vt": (0.0, 0.0, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.0_t0.0_t": (0.0, 0.0, 't', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.0_t0.0_v": (0.0, 0.0, 'v', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.1_t0.1_vt": (0.1, 0.1, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.1_t0.1_t": (0.1, 0.1, 't', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.1_t0.1_v": (0.1, 0.1, 'v', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.5_t0.5_vt": (0.5, 0.5, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.5_t0.5_t": (0.5, 0.5, 't', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.5_t0.5_v": (0.5, 0.5, 'v', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.1_t0.5_vt": (0.1, 0.5, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.5_t0.1_vt": (0.5, 0.1, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.1_t0.3_vt": (0.1, 0.3, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.3_t0.1_vt": (0.3, 0.1, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.3_t0.7_vt": (0.3, 0.7, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
        "T2DR_v0.7_t0.3_vt": (0.7, 0.3, 'vt', "T2DR", weights_T2DR_mvsa, 'Mask'),
    }
    

    test_scenarios = test_t2dr_scenarios_mvsa

    for scenario_name, (v_rate, t_rate, m_rate, model_name, checkpoint_path, intra_missing) in test_scenarios.items():
        print(f"\n{'='*80}")
        print(f"Testing scenario: {scenario_name}")
        print(f"{'='*80}")
        
        args.modelname = model_name
        args.test_checkpoint = checkpoint_path
        args.intra_missing = intra_missing
        if args.dataset == 'MVSA_Single':
            if args.modelname == 'IGML' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.text_missing_rate = t_rate
                val_modality = m_rate
                test_dataset1 = MVSADataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = MVSADataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = MVSADataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = MVSADataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = MVSADataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)

            elif args.modelname == 'TMDC' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.text_missing_rate = t_rate
                val_modality = m_rate
                test_dataset1 = MVSADataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = MVSADataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = MVSADataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = MVSADataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = MVSADataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)

            elif args.modelname == 'T2DR' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.text_missing_rate = t_rate
                val_modality = m_rate
                test_dataset1 = MVSA_t2dr_Dataset(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = MVSA_t2dr_Dataset(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = MVSA_t2dr_Dataset(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = MVSA_t2dr_Dataset(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = MVSA_t2dr_Dataset(args, mode='test', val_modality=val_modality, add_noise=True)


        print(f"Parameters: visual_missing_rate={v_rate}, text_missing_rate={t_rate}, "
              f"val_modality={m_rate}, modelname={model_name}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        

        test_dataloader1 = DataLoader(test_dataset1, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=True)
        test_dataloader2 = DataLoader(test_dataset2, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=True)
        test_dataloader3 = DataLoader(test_dataset3, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=True)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=True)
        test_dataloader5 = DataLoader(test_dataset5, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=True)
        
        if args.modelname == 'IGML':
            args.pe = 1
            if args.dataset in ['MVSA_Single']:
                model = AVClassifier_AUXI_TV(args)

            key_mapping = {
                "fusion_fc.weight": "fusion_module.fc.weight",
                "fusion_fc.bias": "fusion_module.fc.bias",
                "fusion_fc_out.weight": "fusion_module.fc_out.weight",
                "fusion_fc_out.bias": "fusion_module.fc_out.bias"
            }
            state_dict = torch.load(args.test_checkpoint)
            new_state_dict = collections.OrderedDict()
            for k, v in state_dict['model'].items():
                name = k.replace("module.", "") if k.startswith("module.") else k
                if name in key_mapping:
                    final_k = key_mapping[name]
                else:
                    final_k = name
                
                new_state_dict[final_k] = v
            
            model.load_state_dict(new_state_dict, strict=True)
            print("Successfully Loading Weights.......")
        
        elif args.modelname == 'TMDC':
            args.pe = 0
            model = build_model_vt(args)
            state_dict = torch.load(args.test_checkpoint)
            model.load_state_dict(state_dict['model_state_dict'], strict=True)

        elif args.modelname == 'T2DR':
            args.pe = 0
            model = AVClassifier_basic(args)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            with torch.no_grad():
                model.eval()
                
                for batch in test_dataloader1:
                    text_ids, images, labels, text_mask, images_pixel_mask = [
                        x.to(device) for x in batch
                    ]

                    out = model(
                        text_ids,
                        images,
                        text_mask,
                        images_pixel_mask,
                        labels=labels,
                        training=False
                    )
                    break

            state_dict = torch.load(args.test_checkpoint)
            model.load_state_dict(state_dict['model_state_dict'], strict=True)


        model.to(device)
        
        # 测试
        # import ipdb; ipdb.set_trace();
        # print("0")
        total_acc1, f11 = test(args, model, device, test_dataloader1)
        # print("1")
        total_acc2, f12 = test(args, model, device, test_dataloader2)
        total_acc3, f13 = test(args, model, device, test_dataloader3)
        total_acc4, f14 = test(args, model, device, test_dataloader4)
        total_acc5, f15 = test(args, model, device, test_dataloader5)

        total_acc = (total_acc1 + total_acc2 + total_acc3 + total_acc4 + total_acc5) / 5
        f1 = (f11 + f12 + f13 + f14 + f15) / 5
        
        # 输出结果
        print(f"\nResults for {scenario_name}:")
        print(f"  Accuracy: {total_acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Visual Missing Rate: {v_rate}")
        print(f"  Audio Missing Rate: {t_rate}")
        print(f"  Modality Missing Rate: {m_rate}")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()

    
