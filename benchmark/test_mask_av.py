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

from dataset.KSDataset import KSDataset_Noise_Mask, KSDataset_Noise, KSDataset_Noise_tmdc, KSDataset_Noise_tmdc_mask, KSDataset_Noise_t2dr_mask, KSDataset_t2dr_Gaussian
from dataset.CramedDataset import CramedDataset_gaussian, CramedDataset_mask, CramedDataset_tmdc_gaussian, CramedDataset_tmdc_mask, CramedDataset_t2dr_mask, CramedDataset_t2dr_gaussian
from torch.utils.data import DataLoader

import argparse
from models.basic_model import AVClassifier_AUXI_AV
from sklearn.metrics import f1_score 
import numpy as np 

from models.tmdc import build_model_av
from main_t2dr_noise_av import AVClassifier_basic


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='KineticSound', type=str,
                        help='CREMAD, KineticSound, etc.')
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    # 数据路径
    parser.add_argument('--audio_path', default='/root/autodl-tmp/data/CREMA-D/AudioWAV/', type=str)
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

    parser.add_argument('--backbone', default='resnet', type=str, help='where learning rate decays')

    # 噪声参数
    parser.add_argument('--visual_missing_rate', default=0.1, type=float, help='missing rate for visual')
    parser.add_argument('--audio_missing_rate', default=0.1, type=float, help='missing rate for audio')
    parser.add_argument('--modality_missing', default=0.1, type=float, help='missing rate for audio')
    parser.add_argument('--patch_size', default=16, type=int, help='patch size for masking')
    parser.add_argument('--apply_roll', default=False, type=bool, help='apply window shift')
    parser.add_argument('--gamma', type=float, default=4.0)
    parser.add_argument('--beta', type=float, default=1e-5)
    parser.add_argument('--max', type=int, default=1e20)
    parser.add_argument('--drop', default=0, type=int)


    # 保存路径
    parser.add_argument('--ckpt_path', default='/root/autodl-tmp/my/IGML_PE_ARL/results/iadr_new/', type=str, help='path to save trained models')
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
    parser.add_argument('--test_checkpoint', default='/root/autodl-tmp/my/IGML_PE_ARL/oge_results/best_model_of_dataset_CREMAD_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_98_acc_0.7400568181818182.pth', type=str, help='for benchmark')
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
                # CramedDataset_mask
                # spectrogram, images, label, visual_variance, audio_variance
                spec, images, labels, _, _ = data
                spec = spec.to(device)
                images = images.to(device)
                labels = labels.to(device)
                a, v, out, a_feature, v_feature, _, _, _, _, out_a, out_v, a_std_fc, v_std_fc = model(spec.unsqueeze(1).float(), images.float())

                
            elif args.modelname == 'T2DR':
                # CramedDataset_t2dr_mask
                # images, spec, labels, images_pixel_mask, spec_pixel_mask
                images, spec, labels, images_pixel_mask, spec_pixel_mask = data
                spec = spec.to(device)
                images = images.to(device)
                labels = labels.to(device)
                images_pixel_mask = images_pixel_mask.to(device)
                spec_pixel_mask = spec_pixel_mask.to(device)

                out = model(
                    spec.float(), images.float(), spec_pixel_mask, images_pixel_mask,
                    labels=labels, training=False
                )
                    
            elif args.modelname == 'TMDC':
                
                audio = data[0]
                visual = data[1]
                labels = data[2]  
                av_avail = data[3]
                batch_size = audio.shape[0]

                av_avail = torch.stack(av_avail)
                av_avail = torch.transpose(av_avail, 0, 1)
                input_features_mask = av_avail.unsqueeze(1)

                audio = audio.float().to(args.device)
                visual = visual.float().to(args.device)
                input_features_mask = input_features_mask.float().to(args.device)
                labels = labels.long().to(args.device)
                
                # ========== forward ==========
                hidden, out, outputs_intras, outputs_inters, vib_outputs, vib_kls = model(
                    audio, 
                    visual,  
                    input_features_mask,
                    False
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # gpu_ids = list(range(torch.cuda.device_count()))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Gaussian
    ogm_weights_Gaussian = "/root/autodl-tmp/data/results/cramed/best_model_of_dataset_CREMAD_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_85_acc_0.7318548387096775.pth"
    ogm_test_scenarios_Gaussian = {
        "IGML_vclean_aclean_av": ('clean', 'clean', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vclean_aclean_a": ('clean', 'clean', 'a', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vclean_aclean_v": ('clean', 'clean', 'v', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vlow_alow_av": ('low', 'low', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vlow_alow_a": ('low', 'low', 'a', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vlow_alow_v": ('low', 'low', 'v', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vmedium_amedium_av": ('medium', 'medium', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vmedium_amedium_a": ('medium', 'medium', 'a', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vmedium_amedium_v": ('medium', 'medium', 'v', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vlow_amedium_av": ('low', 'medium', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vmedium_alow_a": ('medium', 'low', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vlow_ahigh_v": ('low', 'high', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vhigh_alow_v": ('high', 'low', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vhigh_amedium_v": ('high', 'medium', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
        "IGML_vmedium_ahigh_v": ('medium', 'high', 'av', "IGML", ogm_weights_Gaussian, 'Gaussian'),
    }

    ogm_weights_Gaussian_ks = "/root/autodl-tmp/data/results/ks/best_model_of_dataset_KineticSound_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_97_acc_0.6800367421922842.pth"
    
    ogm_test_scenarios_Gaussian_ks = {
        "IGML_vclean_aclean_av": ('clean', 'clean', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vclean_aclean_a": ('clean', 'clean', 'a', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vclean_aclean_v": ('clean', 'clean', 'v', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vlow_alow_av": ('low', 'low', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vlow_alow_a": ('low', 'low', 'a', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vlow_alow_v": ('low', 'low', 'v', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vmedium_amedium_av": ('medium', 'medium', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vmedium_amedium_a": ('medium', 'medium', 'a', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vmedium_amedium_v": ('medium', 'medium', 'v', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vlow_amedium_av": ('low', 'medium', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vmedium_alow_a": ('medium', 'low', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vlow_ahigh_v": ('low', 'high', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vhigh_alow_v": ('high', 'low', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vhigh_amedium_v": ('high', 'medium', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
        "IGML_vmedium_ahigh_v": ('medium', 'high', 'av', "IGML", ogm_weights_Gaussian_ks, 'Gaussian'),
    }

    
    # IGML
    # mask
    weights_ogm_cramed = "/root/autodl-tmp/data/results/cramed_fanhua/best_model_of_dataset_CREMAD_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_81_acc_0.7184139784946237.pth"
    test_scenarios_ogm_cramed = {
        "IGML_v0.0_a0.0_av": (0.0, 0.0, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.0_a0.0_a": (0.0, 0.0, 'a', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.0_a0.0_v": (0.0, 0.0, 'v', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.1_a0.1_av": (0.1, 0.1, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.1_a0.1_a": (0.1, 0.1, 'a', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.1_a0.1_v": (0.1, 0.1, 'v', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.5_a0.5_av": (0.5, 0.5, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.5_a0.5_a": (0.5, 0.5, 'a', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.5_a0.5_v": (0.5, 0.5, 'v', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.1_a0.5_av": (0.1, 0.5, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.5_a0.1_a": (0.5, 0.1, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.1_a0.3_v": (0.1, 0.3, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.3_a0.1_v": (0.3, 0.1, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.3_a0.7_v": (0.3, 0.7, 'av', "IGML", weights_ogm_cramed, 'Mask'),
        "IGML_v0.7_a0.3_v": (0.7, 0.3, 'av', "IGML", weights_ogm_cramed, 'Mask'),
    }
    # ks
    # weights_ogm_ks = "/root/autodl-tmp/data/results/ks/best_model_of_dataset_KineticSound_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_97_acc_0.6800367421922842.pth"
    # weights_ogm_ks = "/root/autodl-tmp/data/results/ks_fanhua/best_model_of_dataset_KineticSound_Normal_gamma_4.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_80_acc_0.6834047764849969.pth"
    weights_ogm_ks = "/root/autodl-tmp/data/results/ks_gamma/best_model_of_dataset_KineticSound_Normal_gamma_2.0_pe_1_beta1e-05_optimizer_sgd_modulate_starts_0_ends_50_epoch_99_acc_0.6613594611145132.pth"
    test_scenarios_ogm_ks = {
        "IGML_v0.0_a0.0_av": (0.0, 0.0, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.0_a0.0_a": (0.0, 0.0, 'a', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.0_a0.0_v": (0.0, 0.0, 'v', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.1_a0.1_av": (0.1, 0.1, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.1_a0.1_a": (0.1, 0.1, 'a', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.1_a0.1_v": (0.1, 0.1, 'v', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.5_a0.5_av": (0.5, 0.5, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.5_a0.5_a": (0.5, 0.5, 'a', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.5_a0.5_v": (0.5, 0.5, 'v', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.1_a0.5_av": (0.1, 0.5, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.5_a0.1_a": (0.5, 0.1, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.1_a0.3_v": (0.1, 0.3, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.3_a0.1_v": (0.3, 0.1, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.3_a0.7_v": (0.3, 0.7, 'av', "IGML", weights_ogm_ks, 'Mask'),
        "IGML_v0.7_a0.3_v": (0.7, 0.3, 'av', "IGML", weights_ogm_ks, 'Mask'),
    }

    
    # # TMDC
    # # mask
    # # cramed
    # weights_tmdc_cramed = "/root/autodl-tmp/data/results/cramed_tmdc/CREMAD_best_second_stage_acc0.6448863636363636_epochs100_epoch86.pth"
    # test_scenarios_tmdc_cramed = {
    #     "TMDC_v0.0_a0.0_av": (0.0, 0.0, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.0_a0.0_a": (0.0, 0.0, 'a', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.0_a0.0_v": (0.0, 0.0, 'v', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.1_a0.1_av": (0.1, 0.1, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.1_a0.1_a": (0.1, 0.1, 'a', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.1_a0.1_v": (0.1, 0.1, 'v', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.5_a0.5_av": (0.5, 0.5, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.5_a0.5_a": (0.5, 0.5, 'a', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.5_a0.5_v": (0.5, 0.5, 'v', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.1_a0.5_av": (0.1, 0.5, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.5_a0.1_a": (0.5, 0.1, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.1_a0.3_v": (0.1, 0.3, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.3_a0.1_v": (0.3, 0.1, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.3_a0.7_v": (0.3, 0.7, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    #     "TMDC_v0.7_a0.3_v": (0.7, 0.3, 'av', "TMDC", weights_tmdc_cramed, 'Mask'),
    # }
    # # ks
    # weights_tmdc_ks = "/root/autodl-tmp/data/results/ks_tmdc_t2dr/KineticSound_best_second_stage_acc0.650625_epochs100_epoch51.pth"
    # test_scenarios_tmdc_ks = {
    #     "TMDC_v0.0_a0.0_av": (0.0, 0.0, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.0_a0.0_a": (0.0, 0.0, 'a', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.0_a0.0_v": (0.0, 0.0, 'v', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.1_a0.1_av": (0.1, 0.1, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.1_a0.1_a": (0.1, 0.1, 'a', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.1_a0.1_v": (0.1, 0.1, 'v', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.5_a0.5_av": (0.5, 0.5, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.5_a0.5_a": (0.5, 0.5, 'a', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.5_a0.5_v": (0.5, 0.5, 'v', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.1_a0.5_av": (0.1, 0.5, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.5_a0.1_a": (0.5, 0.1, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.1_a0.3_v": (0.1, 0.3, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.3_a0.1_v": (0.3, 0.1, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.3_a0.7_v": (0.3, 0.7, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    #     "TMDC_v0.7_a0.3_v": (0.7, 0.3, 'av', "TMDC", weights_tmdc_ks, 'Mask'),
    # }

    # # mask
    # # cramed
    # weights_t2dr_cramed = "/root/autodl-tmp/data/results/cramed_t2dr/final_best_model_CREMAD_epoch87_acc0.6576.pth"
    # test_scenarios_t2dr_cramed = {
    #     "T2DR_v0.0_a0.0_av": (0.0, 0.0, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.0_a0.0_a": (0.0, 0.0, 'a', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.0_a0.0_v": (0.0, 0.0, 'v', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.1_a0.1_av": (0.1, 0.1, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.1_a0.1_a": (0.1, 0.1, 'a', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.1_a0.1_v": (0.1, 0.1, 'v', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.5_a0.5_av": (0.5, 0.5, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.5_a0.5_a": (0.5, 0.5, 'a', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.5_a0.5_v": (0.5, 0.5, 'v', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.1_a0.5_av": (0.1, 0.5, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.5_a0.1_a": (0.5, 0.1, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.1_a0.3_v": (0.1, 0.3, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.3_a0.1_v": (0.3, 0.1, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.3_a0.7_v": (0.3, 0.7, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    #     "T2DR_v0.7_a0.3_v": (0.7, 0.3, 'av', "T2DR", weights_t2dr_cramed, 'Mask'),
    # }
    # # ks
    # weights_t2dr_ks = "/root/autodl-tmp/data/results/ks_t2dr_fanhua/final_best_model_KineticSound_epoch80_acc0.6528.pth"
    # test_scenarios_t2dr_ks = {
    #     # "T2DR_v0.0_a0.0_av": (0.0, 0.0, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     # "T2DR_v0.0_a0.0_a": (0.0, 0.0, 'a', "T2DR", weights_t2dr_ks, 'Mask'),
    #     # "T2DR_v0.0_a0.0_v": (0.0, 0.0, 'v', "T2DR", weights_t2dr_ks, 'Mask'),
    #     # "T2DR_v0.1_a0.1_av": (0.1, 0.1, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.1_a0.1_a": (0.1, 0.1, 'a', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.1_a0.1_v": (0.1, 0.1, 'v', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.5_a0.5_av": (0.5, 0.5, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.5_a0.5_a": (0.5, 0.5, 'a', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.5_a0.5_v": (0.5, 0.5, 'v', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.1_a0.5_av": (0.1, 0.5, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.5_a0.1_a": (0.5, 0.1, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.1_a0.3_v": (0.1, 0.3, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.3_a0.1_v": (0.3, 0.1, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.3_a0.7_v": (0.3, 0.7, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    #     "T2DR_v0.7_a0.3_v": (0.7, 0.3, 'av', "T2DR", weights_t2dr_ks, 'Mask'),
    # }
    
    test_scenarios = test_scenarios_ogm_ks

    for scenario_name, (v_rate, a_rate, m_rate, model_name, checkpoint_path, intra_missing) in test_scenarios.items():
        print(f"\n{'='*80}")
        print(f"Testing scenario: {scenario_name}")
        print(f"{'='*80}")
        
        args.modelname = model_name
        args.test_checkpoint = checkpoint_path
        args.intra_missing = intra_missing
        if args.dataset == 'KineticSound':
            if args.modelname == 'IGML' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = KSDataset_Noise_Mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset2 = KSDataset_Noise_Mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset3 = KSDataset_Noise_Mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset4 = KSDataset_Noise_Mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset5 = KSDataset_Noise_Mask(args, mode='testt', val_modality=val_modality, add_noise=True)
            if args.modelname == 'IGML' and args.intra_missing == 'Gaussian':
                args.a_noise_intensity_level = v_rate
                args.v_noise_intensity_level = a_rate
                val_modality = m_rate
                test_dataset1 = KSDataset_Noise(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset2 = KSDataset_Noise(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset3 = KSDataset_Noise(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset4 = KSDataset_Noise(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset5 = KSDataset_Noise(args, mode='testt', val_modality=val_modality, add_noise=True)

            if args.modelname == 'TMDC' and args.intra_missing == 'Gaussian':
                args.a_noise_intensity_level = v_rate
                args.v_noise_intensity_level = a_rate
                val_modality = m_rate
                test_dataset1 = KSDataset_Noise_tmdc(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset2 = KSDataset_Noise_tmdc(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset3 = KSDataset_Noise_tmdc(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset4 = KSDataset_Noise_tmdc(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset5 = KSDataset_Noise_tmdc(args, mode='testt', val_modality=val_modality, add_noise=True)
            if args.modelname == 'TMDC' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = KSDataset_Noise_tmdc_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset2 = KSDataset_Noise_tmdc_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset3 = KSDataset_Noise_tmdc_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset4 = KSDataset_Noise_tmdc_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset5 = KSDataset_Noise_tmdc_mask(args, mode='testt', val_modality=val_modality, add_noise=True)

            if args.modelname == 'T2DR' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = KSDataset_Noise_t2dr_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset2 = KSDataset_Noise_t2dr_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset3 = KSDataset_Noise_t2dr_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset4 = KSDataset_Noise_t2dr_mask(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset5 = KSDataset_Noise_t2dr_mask(args, mode='testt', val_modality=val_modality, add_noise=True)

            if args.modelname == 'T2DR' and args.intra_missing == 'Gaussian':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = KSDataset_t2dr_Gaussian(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset2 = KSDataset_t2dr_Gaussian(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset3 = KSDataset_t2dr_Gaussian(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset4 = KSDataset_t2dr_Gaussian(args, mode='testt', val_modality=val_modality, add_noise=True)
                test_dataset5 = KSDataset_t2dr_Gaussian(args, mode='testt', val_modality=val_modality, add_noise=True)

        elif args.dataset == 'CREMAD':
            if args.modelname == 'IGML' and args.intra_missing == 'Gaussian':
                args.a_noise_intensity_level = v_rate
                args.v_noise_intensity_level = a_rate
                val_modality = m_rate
                test_dataset1 = CramedDataset_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = CramedDataset_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = CramedDataset_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = CramedDataset_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = CramedDataset_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
            if args.modelname == 'IGML' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = CramedDataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = CramedDataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = CramedDataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = CramedDataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = CramedDataset_mask(args, mode='test', val_modality=val_modality, add_noise=True)

            if args.modelname == 'TMDC' and args.intra_missing == 'Gaussian':
                args.a_noise_intensity_level = v_rate
                args.v_noise_intensity_level = a_rate
                val_modality = m_rate
                test_dataset1 = CramedDataset_tmdc_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = CramedDataset_tmdc_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = CramedDataset_tmdc_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = CramedDataset_tmdc_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = CramedDataset_tmdc_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
            if args.modelname == 'TMDC' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = CramedDataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = CramedDataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = CramedDataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = CramedDataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = CramedDataset_tmdc_mask(args, mode='test', val_modality=val_modality, add_noise=True)

            if args.modelname == 'T2DR' and args.intra_missing == 'Mask':
                args.visual_missing_rate = v_rate
                args.audio_missing_rate = a_rate
                val_modality = m_rate
                test_dataset1 = CramedDataset_t2dr_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = CramedDataset_t2dr_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = CramedDataset_t2dr_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = CramedDataset_t2dr_mask(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = CramedDataset_t2dr_mask(args, mode='test', val_modality=val_modality, add_noise=True)

            if args.modelname == 'T2DR' and args.intra_missing == 'Gaussian':
                args.v_noise_intensity_level = v_rate
                args.a_noise_intensity_level = a_rate
                val_modality = m_rate
                test_dataset1 = CramedDataset_t2dr_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset2 = CramedDataset_t2dr_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset3 = CramedDataset_t2dr_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset4 = CramedDataset_t2dr_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)
                test_dataset5 = CramedDataset_t2dr_gaussian(args, mode='test', val_modality=val_modality, add_noise=True)


        print(f"Parameters: visual_missing_rate={v_rate}, audio_missing_rate={a_rate}, "
              f"val_modality={m_rate}, modelname={model_name}")
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found: {checkpoint_path}")
            continue
        

        test_dataloader1 = DataLoader(test_dataset1, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=False)
        test_dataloader2 = DataLoader(test_dataset2, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=False)
        test_dataloader3 = DataLoader(test_dataset3, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=False)
        test_dataloader4 = DataLoader(test_dataset4, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=False)
        test_dataloader5 = DataLoader(test_dataset5, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=0, 
                                     pin_memory=False, drop_last=False)
        
        if args.modelname == 'IGML':
            args.pe = 1
            model = AVClassifier_AUXI_AV(args)

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
            
            # model.load_state_dict(new_state_dict, strict=True)
            model.load_state_dict(new_state_dict, strict=False)
            print("Successfully Loading Weights.......")
        
        elif args.modelname == 'TMDC':
            args.pe = 0
            model = build_model_av(args)
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
                    images, spec, labels, images_pixel_mask, spec_pixel_mask = [
                        x.to(device) for x in batch
                    ]
                    _ = model(
                        spec.float(), images.float(), 
                        spec_pixel_mask, images_pixel_mask,
                        labels=labels, training=False
                    )
                    break 

            state_dict = torch.load(args.test_checkpoint)
            model.load_state_dict(state_dict['model_state_dict'], strict=True)
            # model.load_state_dict(state_dict['model_state_dict'], strict=False)

        # model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        model.to(device)
        
        # 测试
        # import ipdb; ipdb.set_trace();
        print("0")
        total_acc1, f11 = test(args, model, device, test_dataloader1)
        print("1")
        total_acc2, f12 = test(args, model, device, test_dataloader2)
        print("2")
        total_acc3, f13 = test(args, model, device, test_dataloader3)
        print("3")
        total_acc4, f14 = test(args, model, device, test_dataloader4)
        print("4")
        total_acc5, f15 = test(args, model, device, test_dataloader5)
        print("5")

        total_acc = (total_acc1 + total_acc2 + total_acc3 + total_acc4 + total_acc5) / 5
        f1 = (f11 + f12 + f13 + f14 + f15) / 5
        
        # 输出结果
        print(f"\nResults for {scenario_name}:")
        print(f"  Accuracy: {total_acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Visual Missing Rate: {v_rate}")
        print(f"  Audio Missing Rate: {a_rate}")
        print(f"  Modality Missing Rate: {m_rate}")
        
        # 清理内存
        del model
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()

    