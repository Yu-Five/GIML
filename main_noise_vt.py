import argparse
import os
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import ipdb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.MVSADataset import MVSADataset_mask
from models.basic_model import AVClassifier_AUXI_TV
from utils.utils import setup_seed, weight_init
import csv
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    # parser.add_argument('--use_video_frames', default=3, type=int)
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    parser.add_argument('--audio_path', default='/root/autodl-tmp/data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/root/autodl-tmp/data/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max_length', default=128, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--modulation_starts', default=0, type=int, help='where modulation begins')
    parser.add_argument('--modulation_ends', default=50, type=int, help='where modulation ends')
    parser.add_argument('--alpha', required=True, type=float, help='alpha in OGM-GE')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--max', type=int, default=1e20)
    parser.add_argument('--modality', type=str, default='full')
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--drop', default=0, type=int)
    parser.add_argument('--cureent_epoch', default=0, type=int)
    parser.add_argument('--cylcle_epoch', default=40, type=int)
    return parser.parse_args()


def get_feature_diversity(a_feature):
    a_feature = a_feature.view(a_feature.shape[0], a_feature.shape[1], -1)  # B C HW
    a_feature = a_feature.permute(0, 2, 1)  # B HW C
    a_feature = a_feature - torch.mean(a_feature, dim=2, keepdim=True)
    a_similarity = torch.bmm(a_feature, a_feature.permute(0, 2, 1))
    a_std = torch.std(a_feature, dim=2)
    a_std_matrix = torch.bmm(a_std.unsqueeze(dim=2), a_std.unsqueeze(dim=1))
    a_similarity = a_similarity / a_std_matrix
    # print(a_similarity)
    a_norm = torch.norm(a_similarity, dim=(1, 2)) / (a_similarity.shape[1] ** 2)
    # print(a_norm.shape)
    a_norm = torch.mean(a_norm)
    return a_norm


def regurize(mul, std, target_var=2):  # 新增目标方差参数，默认为2
    variance_dul = std ** 2  # 模型预测的方差 σ²
    variance_dul = variance_dul.view(variance_dul.shape[0], -1)
    mul = mul.view(mul.shape[0], -1)  # 模型预测的均值 μ
    
    # print(mul.shape,variance_dul.shape)
    target_var=torch.unsqueeze(target_var,dim=1).cuda()
    # 计算KL散度，使用目标方差target_var（这里为2）
    loss_kl = ( (variance_dul / target_var) + (mul **2 / target_var) 
              - torch.log( (variance_dul + 1e-8) / target_var ) 
              - 1 ) * 0.5
    
    loss_kl = torch.sum(loss_kl, dim=1)  # 对特征维度求和
    loss_kl = torch.mean(loss_kl)        # 对样本求平均
    
    return loss_kl



def get_feature_diff(x1, x2):
    # print(x1.shape,x2.shape)
    x1 = F.adaptive_avg_pool2d(x1, (7, 7))
    x2 = F.adaptive_avg_pool2d(x2, (7, 7))
    # x1 = torch.mean(x1, dim=(2, 3))
    # x2 = torch.mean(x2, dim=(2, 3))

    x1 = x1.permute(0, 2, 3, 1).contiguous()
    x2 = x2.permute(0, 2, 3, 1).contiguous()

    rgb = x1.view(-1, x1.shape[3])
    depth = x2.view(-1, x2.shape[3])

    diff = F.mse_loss(rgb, depth)
    # diff = torch.cosine_similarity(rgb, depth)
    # diff = torch.mean(diff)
    # print(simi.shape)
    return diff


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler,scheduler_warmup=None,
                writer=None):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    if scheduler_warmup is not None:
        scheduler_warmup.step(epoch=epoch + 1)
    elif scheduler is not None:
        print(scheduler)
        scheduler.step(epoch)

    # if epoch < 20:
    print(epoch, optimizer.param_groups[0]['lr'])

    model.train()
    print("Start training ... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _a_diveristy = 0
    _v_diveristy = 0
    _a_re = 0
    _v_re = 0
    similar_average = 0

    model.module.args.current_epoch=epoch

    for step, data in enumerate(tqdm(dataloader, desc="Epoch {}/{}".format(epoch, args.epochs))):
        text_data, image, label,v_variance,t_variance = data
        text_data = {k: v.to(device) for k, v in text_data.items()}
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, out_a, out_v,a_std_fc,v_std_fc = model(text_data, image.float())
        
        # print(a_feature.shape,v_feature.shape)

        # similar = get_feature_diff(a_feature, v_feature)
        similar_average += 0
        # print(similar.mean())

        loss_v = criterion(out_v, label)
        loss_a = criterion(out_a, label)
        loss_f = criterion(out, label)

        calculate_a = torch.mean(torch.abs(out_a), 0).sum().cpu().detach()
        calculate_b = torch.mean(torch.abs(out_v), 0).sum().cpu().detach()

        # print(out_a.shape,out_v.shape)



        loss_cls = loss_f + (loss_a  + loss_v) * args.gamma

        a_diveristy = get_feature_diversity(a_feature)
        v_diveristy = get_feature_diversity(v_feature)

        # if epoch<2:
        #     a_std = torch.clamp(a_std, min=0, max=2)
        #     v_std = torch.clamp(v_std, min=0, max=2)

        # print(a_mul)



        if not isinstance(a_mul, int):
            regurize_a = regurize(a_mul, a_std,target_var=t_variance)
            regurize_a = regurize_a.cuda()
        else:
            regurize_a = torch.zeros(1).float().cuda()
            a_std = torch.zeros(1).float().cuda()

        if not isinstance(v_mul, int):
            if args.num_frame>1:
                v_variance_kl=torch.repeat_interleave(v_variance,args.num_frame)
            else:
                v_variance_kl=v_variance
            regurize_v =  regurize(v_mul, v_std,target_var=v_variance_kl)
            regurize_v = regurize_v.cuda()
        else:
            regurize_v = torch.zeros(1).float().cuda()
            v_std = torch.zeros(1).float().cuda()

        # if epoch < 2:
        #     regurize_loss = torch.zeros(1).float().cuda()
        # else:
        #     regurize_loss = (regurize_a + regurize_v) * args.beta

        regurize_loss = (regurize_a + regurize_v)
        # regurize_loss = (regurize_a * 100 + regurize_v)
        # if regurize_loss>10:
        #     regurize_loss=regurize_loss/(regurize_loss/10.0)
        v_variance=torch.unsqueeze(v_variance.float(),dim=1).cuda()
        t_variance=torch.unsqueeze(t_variance.float(),dim=1).cuda()
        
        variance_fc_loss=F.mse_loss(a_std_fc,t_variance)+F.mse_loss(v_std_fc,v_variance)
        if variance_fc_loss==torch.inf:
            variance_fc_loss=torch.zeros(1).float().cuda()

        # print(variance_fc_loss)

        loss = loss_cls + regurize_loss * args.beta + variance_fc_loss*0.1
        # print(loss)
        if step % 100 == 0:
            # print(a_std.mean().item(),v_std.mean().item())
            print("regurize_Loss:", regurize_loss.item(), "unimodal_loss:", (loss_a + loss_v).item(), "cls_loss:",
                  loss_cls.item(), "var_loss:", variance_fc_loss.item())



        if step % 100 == 0:
            print("calculate:", calculate_a, calculate_b)
            print("variance:",a_std.mean().item(),v_std.mean().item(),a_std_fc.mean().item(),v_std_fc.mean().item(),t_variance.mean(),v_variance.mean())

        #     # print(a.shape, v.shape)
        #     selected_rows_a = torch.index_select(model.module.fusion_module.fc_out.weight[:, :512], dim=0, index=label)
        #     selected_rows_v = torch.index_select(model.module.fusion_module.fc_out.weight[:, 512:], dim=0, index=label)
        #     distance_a = torch.abs(F.cosine_similarity(a, selected_rows_a, dim=1).mean())
        #     distance_v = torch.abs(F.cosine_similarity(v, selected_rows_v, dim=1).mean())
        #     print("distance:", distance_a.item(), distance_v.item())
        #
        # fc_weight = model.module.fusion_module.fc_out.weight
        # fc_weight = fc_weight.T
        #
        # fc_weight_mean = fc_weight[:, 3]
        #
        # visual = torch.mean(torch.abs(fc_weight_mean[0:512]))
        # audio = torch.mean(torch.abs(fc_weight_mean[512:1024]))
        #
        # print("weight:", torch.sum(audio).cpu().detach().numpy(), torch.sum(visual).cpu().detach().numpy())

        # with open("weight_of_a_v.csv", 'a', newline='') as f:
        #     writer = csv.writer(f)
        #     row = [calculate_a, calculate_b]
        #     writer.writerow(row)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)



        # if (acc_v + acc_a) != 0:
        #
        #     if (calculate_a / calculate_b) < (acc_a / acc_v):
        #         if step % 20 == 0:
        #             print("calculate smaller than target", (acc_a / calculate_a) * (calculate_b / acc_v))
        #         for p in model.module.audio_net.parameters():
        #             p.grad = p.grad * (acc_a / calculate_a) * (calculate_b / acc_v)
        #     else:
        #         for p in model.module.visual_net.parameters():
        #             if step % 20 == 0:
        #                 print("calculate large than target",(calculate_a / acc_a) * (acc_v / calculate_b))
        #             p.grad = p.grad * (calculate_a / acc_a) * (acc_v / calculate_b)


        text_grad_sum = 0
        index=0
        for p in model.module.text_net.parameters():
            index+=1
            # print(p.grad)
            if p.grad is not None:
                text_grad_sum += torch.abs(p.grad).mean().item()

        visual_grad_sum = 0
        index=0
        for p in model.module.visual_net.parameters():
            index+=1
            visual_grad_sum += torch.abs(p.grad).mean().item()
        if step % 100 == 0:
            print("grad:",text_grad_sum, visual_grad_sum)

        file_name = 'text_visual_grad_vanilla' + '.csv'
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [text_grad_sum, visual_grad_sum]
            writer.writerow(row)
        
        
        

        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _a_diveristy += a_diveristy.item()
        _v_diveristy += v_diveristy.item()
        _a_re += regurize_a.item()
        _v_re += regurize_v.item()

        # if step % 100 == 0:
        #     print(step, loss)

    similar_average = similar_average / (step + 1)
    print("mse_diff:", similar_average)
    # print(regurize_v,regurize_a)
    # file_name = 'audio_visual_similar_in_numtimodal' + '.csv'
    # with open(file_name, 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     row = [similar_average.cpu().detach().numpy()]
    #     writer.writerow(row)
    print(_loss,len(dataloader))
    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_diveristy / len(
        dataloader), _v_diveristy / len(dataloader), _a_re / len(dataloader), _v_re / len(dataloader),



def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'MVSA_Single':
        n_classes = 3

    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    model.module.args.drop = 0
    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        # print(model.module.args.drop)
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (spec, image, label,t_variance,v_variance) in enumerate(dataloader):
            spec = {k: v.to(device) for k, v in spec.items()}
            image = image.to(device)
            label = label.to(device)

            # TODO: make it simpler and easier to extend
            a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, out_a, out_v,a_std_fc,v_std_fc = model(spec, image.float())


            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(image.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

    # model.module.args.drop = 1
    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)



def main():
    args = get_arguments()
    args.p = [0, 0]
    print(args)

    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')


    model = AVClassifier_AUXI_TV(args)
    # model.apply(weight_init)
    # for name, param in model.text_net.named_parameters():
    #     if "encoder.layer.0" in name \
    #         or "encoder.layer.1" in name \
    #         or "encoder.layer.2" in name \
    #         or "embeddings" in name:
    #         param.requires_grad = False

    if args.pretrain:
        print("no init, load pretrain!")
        for name, module in model.named_children():
            if name not in ['visual_net', 'text_net']:
                print(name)
                module.apply(weight_init)
    else: 
        for name, module in model.named_children():
            if name not in ['text_net']:
                print(name)
                module.apply(weight_init)

    for name, param in model.text_net.named_parameters():
        if "embeddings" in name:
            param.requires_grad = False
        
        for i in range(9): 
            if f"encoder.layer.{i}." in name:
                param.requires_grad = False

    model.to(device)

    model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    model.cuda()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, eval(args.lr_decay_step), args.lr_decay_ratio)
        # 创建学习率调度器
        # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    elif args.optimizer == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
        scheduler = None
    elif args.optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = None
    else:
        raise ValueError

    # if args.warmup:
    #     scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
    #                                               after_scheduler=scheduler)
    # else:
    scheduler_warmup = None


    if args.dataset == 'MVSA_Single':
        train_dataset_noise = MVSADataset_mask(args, mode='train',add_noise=True)
        test_dataset = MVSADataset_mask(args, mode='valid',add_noise=False)
        test_dataset_noise5 = MVSADataset_mask(args, mode='valid', add_noise=True)
    
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound, KineticSound and CREMA-D for now!'.format(args.dataset))

    # train_dataloader_clean = DataLoader(train_dataset_clean, batch_size=args.batch_size,
    #                               shuffle=True, num_workers=32, pin_memory=True, drop_last = False)

    train_dataloader_noise = DataLoader(train_dataset_noise, batch_size=args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True, drop_last = False)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last = False)
    test_dataloader_noise5 = DataLoader(test_dataset_noise5, batch_size=args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last = False)
    # test_dataloader_noisehalf = DataLoader(test_dataset_noisehalf, batch_size=args.batch_size,
    #                              shuffle=False, num_workers=32, pin_memory=True, drop_last = False)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    log_path = os.path.join(args.ckpt_path, 'Noise_' + args.dataset + 'gamma' + str(args.gamma) + '_'  + args.fusion_method + '_' + args.modality + '_' + '.csv')
    print(log_path)
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([1000, 1000, 1000])
    if args.train:
        acc = 0
        acc5, acc_a5, acc_v5 = 0, 0, 0
        acc_clean, acc_a_clean, acc_v_clean = 0, 0, 0
        # acc_noisehalf, acc_a_noisehalf, acc_v_noisehalf = 0, 0, 0
        best_epoch = 0
        best_acc = 0.0
        saved_dict = {'saved_epoch': 0,
                    'modulation': None,
                    'alpha': 0,
                    'fusion': None,
                    'acc': 0,
                    'model': None,
                    'optimizer': None,
                    }

        for epoch in range(args.epochs):

            # if epoch<args.cylcle_epoch:
            #     train_dataloader=train_dataloader_clean
            # else:
            print("Always train_dataset_noise")
            train_dataloader=train_dataloader_noise

            print('Epoch: {}: '.format(epoch))
            args.epoch_now = epoch

            if args.use_tensorboard:

                writer_path = os.path.join(args.tensorboard_path, args.dataset)
                if not os.path.exists(writer_path):
                    os.mkdir(writer_path)
                log_name = '{}_{}'.format(args.fusion_method, args.modulation)
                writer = SummaryWriter(os.path.join(writer_path, log_name))

                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args,
                                                                                                           epoch,
                                                                                                           model,
                                                                                                           device,
                                                                                                           train_dataloader,
                                                                                                           optimizer,
                                                                                                           scheduler,
                                                                                                           scheduler_warmup, )
                
                acc, acc_a, acc_v = valid(args, model, device, test_dataloader)

                writer.add_scalars('Loss', {'Total Loss': batch_loss,
                                            'Audio Loss': batch_loss_a,
                                            'Visual Loss': batch_loss_v}, epoch)

                writer.add_scalars('Evaluation', {'Total Accuracy': acc,
                                                  'Audio Accuracy': acc_a,
                                                  'Visual Accuracy': acc_v}, epoch)

            else:
                batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args=args, epoch=epoch,
                                                                                                           model=model,
                                                                                                           device=device,
                                                                                                           dataloader=train_dataloader,
                                                                                                           optimizer=optimizer,
                                                                                                           scheduler=scheduler
                                                                                                           )
                # import ipdb; ipdb.set_trace();
                # if epoch > (args.epochs) // 2: 
                # schuduler是在0.7上，那就0.7之后在保存吧，都改成这样的
                if epoch > (args.epochs) * 0.7:
                    acc5, acc_a5, acc_v5 = valid(args, model, device, test_dataloader_noise5)
                    acc_clean, acc_a_clean, acc_v_clean = valid(args, model, device, test_dataloader)
                    # acc_noisehalf, acc_a_noisehalf, acc_v_noisehalf = valid(args, model, device, test_dataloader_noisehalf)
                    acc = (acc5 + acc_clean) / 2

                print(11111111111)
                with open(log_path, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow([acc, acc5, acc_a5, acc_v5, acc_clean, acc_a_clean, acc_v_clean])

            if acc > best_acc:
                best_acc = float(acc)
                best_epoch = epoch
                saved_dict = {'saved_epoch': epoch,
                                'modulation': args.modulation,
                                'alpha': args.alpha,
                                'fusion': args.fusion_method,
                                'acc': best_acc,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                }

            print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
            print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
            print("Audio regurize: {:.3f}， Visual regurize: {:.3f} ".format(a_re, v_re))


        if not os.path.exists(args.ckpt_path):
            os.makedirs(args.ckpt_path)

        model_name = 'best_model_of_dataset_{}_{}_gamma_{}_pe_{}_beta{}_' \
                        'optimizer_{}_modulate_starts_{}_ends_{}_' \
                        'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                    args.modulation,
                                                    args.gamma,
                                                    args.pe,
                                                    args.beta,
                                                    args.optimizer,
                                                    args.modulation_starts,
                                                    args.modulation_ends,
                                                    best_epoch, best_acc)

        save_dir = os.path.join(args.ckpt_path, model_name)

        torch.save(saved_dict, save_dir)
        print('The best model has been saved at {}.'.format(save_dir))

    # else:
    #     # first load trained model
    #     loaded_dict = torch.load("results/ave/full_nosie_weight_train_res_cycle50_variance2_noise_1e-1_nodrop/best_model_of_dataset_AVE_Normal_gamma_2.5_pe_1_beta0.0_optimizer_sgd_modulate_starts_0_ends_50_epoch_77_acc_0.6796875.pth")
    #     # epoch = loaded_dict['saved_epoch']
    #     modulation = loaded_dict['modulation']
    #     # alpha = loaded_dict['alpha']
    #     fusion = loaded_dict['fusion']
    #     state_dict = loaded_dict['model']
    #     # optimizer_dict = loaded_dict['optimizer']
    #     # scheduler = loaded_dict['scheduler']

    #     assert modulation == args.modulation, 'inconsistency between modulation method of loaded model and args !'
    #     assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'
    #     # print(state_dict)
    #     model.load_state_dict(state_dict)
    #     # model.train()
    #     # model.eval()
    #     print('Trained model loaded!')

    #     acc, acc_a, acc_v = valid(args, model, device, test_dataloader)
    #     print('Accuracy: {}, accuracy_a: {}, accuracy_v: {}'.format(acc, acc_a, acc_v))


if __name__ == "__main__":
    main()


