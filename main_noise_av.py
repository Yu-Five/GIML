import argparse
import os
import pstats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pdb
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.CramedDataset import CramedDataset_mask
from models.basic_model import AVClassifier_AUXI_AV, AVClassifier_AUXI_AV_NoDecouple
from utils.utils import setup_seed, weight_init
from dataset.KSDataset import KSDataset_Noise_Mask
import csv
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--num_frame', default=3, type=int, help='use how many frames for train')

    parser.add_argument('--audio_path', default='/root/autodl-fs/CREMA-D-Only/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/root/autodl-fs/CREMA-D-Only/CREMA-D', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max_length', default=128, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default='[70]', type=str, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')


    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='1', type=str, help='GPU ids')
    parser.add_argument('--pe', type=int, default=0)
    parser.add_argument('--max', type=int, default=1e20)
    parser.add_argument('--beta', type=float, default=0)

    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument('--total_epoch', default=10, type=int)
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--drop', default=0, type=int)
    parser.add_argument('--cureent_epoch', default=0, type=int)
    parser.add_argument('--cylcle_epoch', default=10, type=int)

    parser.add_argument('--no_decouple', default=0, type=int)


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
        # pdb.set_trace()
        spec, image, label,v_variance,a_variance = data
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, out_a, out_v,a_std_fc,v_std_fc = model(spec.unsqueeze(1).float(),
                                                                                            image.float())
            
        
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
            regurize_a = regurize(a_mul, a_std,target_var=a_variance)
            regurize_a = regurize_a.cuda()
        else:
            regurize_a = torch.zeros(1).float().cuda()
            a_std = torch.zeros(1).float().cuda()

        if not isinstance(v_mul, int):
            if args.num_frame>1:
                v_variance_kl=torch.repeat_interleave(v_variance,args.num_frame)
            else:
                v_variance_kl=v_variance
            # import ipdb; ipdb.set_trace();
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
        a_variance=torch.unsqueeze(a_variance.float(),dim=1).cuda()
        
        variance_fc_loss=F.mse_loss(a_std_fc,a_variance)+F.mse_loss(v_std_fc,v_variance)
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
            print("variance:",a_std.mean().item(),v_std.mean().item(),a_std_fc.mean().item(),v_std_fc.mean().item(),a_variance.mean(),v_variance.mean())


        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        
        audio_grad_sum = 0
        index=0
        for p in model.module.audio_net.parameters():
            index+=1
            # print(p.grad)
            audio_grad_sum += torch.abs(p.grad).mean().item()

        visual_grad_sum = 0
        index=0
        for p in model.module.visual_net.parameters():
            index+=1
            visual_grad_sum += torch.abs(p.grad).mean().item()
        if step % 100 == 0:
            print("grad:",audio_grad_sum, visual_grad_sum)

        file_name = 'audio_visual_grad_vanilla' + '.csv'
        with open(file_name, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [audio_grad_sum, visual_grad_sum]
            writer.writerow(row)



        optimizer.step()

        _loss += loss.item()
        _loss_a += loss_a.item()
        _loss_v += loss_v.item()
        _a_diveristy += a_diveristy.item()
        _v_diveristy += v_diveristy.item()
        _a_re += regurize_a.item()
        _v_re += regurize_v.item()


    similar_average = similar_average / (step + 1)
    print("mse_diff:", similar_average)

    print(_loss,len(dataloader))
    return _loss / len(dataloader), _loss_a / len(dataloader), _loss_v / len(dataloader), _a_diveristy / len(
        dataloader), _v_diveristy / len(dataloader), _a_re / len(dataloader), _v_re / len(dataloader),



def valid(args, model, device, dataloader):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'KineticSound':
        n_classes = 34
    elif args.dataset == 'CREMAD':
        n_classes = 6

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

        for step, (spec, image, label,a_variance,v_variance) in enumerate(dataloader):
            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out, a_feature, v_feature, _, _, _, _, out_a, out_v,a_std_fc,v_std_fc = model(spec.unsqueeze(1).float(), image.float())

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

    if args.no_decouple == 0:
        print("decouple")
        model = AVClassifier_AUXI_AV(args)
    else: 
        print("no_decouple")
        model = AVClassifier_AUXI_AV_NoDecouple(args)

    model.apply(weight_init)


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

    scheduler_warmup = None

    if args.dataset == 'KineticSound':
        train_dataset_noise = KSDataset_Noise_Mask(args, mode='train',add_noise=True)
        test_dataset = KSDataset_Noise_Mask(args, mode='test',add_noise=False)
        test_dataset_noise5 = KSDataset_Noise_Mask(args, mode='test',add_noise=True, val_half=False)
        # test_dataset_noisehalf = KSDataset_Noise_Mask(args, mode='test',add_noise=True, val_half=True)
    elif args.dataset == 'CREMAD':
        train_dataset_noise = CramedDataset_mask(args, mode='train',add_noise=True)
        test_dataset = CramedDataset_mask(args, mode='valid',add_noise=False)
        test_dataset_noise5 = CramedDataset_mask(args, mode='valid',add_noise=True, val_half=False)
        # test_dataset_noisehalf = CramedDataset_mask(args, mode='valid',add_noise=True, val_half=True)
    
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
    log_path = os.path.join(args.ckpt_path, 'Noise_' + args.dataset + 'gamma' + str(args.gamma) + 'concat' + '_' + '.csv')
    print(log_path)

    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([1000, 1000, 1000])



    acc = 0
    acc5, acc_a5, acc_v5 = 0, 0, 0
    acc_clean, acc_a_clean, acc_v_clean = 0, 0, 0
    acc_noisehalf, acc_a_noisehalf, acc_v_noisehalf = 0, 0, 0
    best_epoch = 0
    best_acc = 0.0
    saved_dict = {'saved_epoch': 0,
                'acc': 0,
                'model': None,
                'optimizer': None,
                }

    for epoch in range(args.epochs):

        print("Always train_dataset_noise")
        train_dataloader=train_dataloader_noise

        print('Epoch: {}: '.format(epoch))
        args.epoch_now = epoch


        batch_loss, batch_loss_a, batch_loss_v, a_diveristy, v_diveristy, a_re, v_re = train_epoch(args=args, epoch=epoch,
                                                                                                    model=model,
                                                                                                    device=device,
                                                                                                    dataloader=train_dataloader,
                                                                                                    optimizer=optimizer,
                                                                                                    scheduler=scheduler
                                                                                                    )

        # import ipdb; ipdb.set_trace();
        # schuduler是在0.7上，那就0.7之后在保存吧，都改成这样的
        if epoch > (args.epochs) * 0.7:
            acc5, acc_a5, acc_v5 = valid(args, model, device, test_dataloader_noise5)
            acc_clean, acc_a_clean, acc_v_clean = valid(args, model, device, test_dataloader)
            # acc_noisehalf, acc_a_noisehalf, acc_v_noisehalf = valid(args, model, device, test_dataloader_noisehalf)
            acc = (acc5 + acc_clean) / 2

        print(11111111111)
        with open(log_path, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([acc, acc5, acc_a5, acc_v5, acc_noisehalf, acc_a_noisehalf, acc_v_noisehalf, acc_clean, acc_a_clean, acc_v_clean])

        if acc > best_acc:
            best_acc = float(acc)
            best_epoch = epoch
            saved_dict = {'saved_epoch': epoch,
                            'acc': best_acc,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }

        print("Loss: {:.3f}, Acc: {:.3f}, Best Acc: {:.3f}".format(batch_loss, acc, best_acc))
        print("Audio similar: {:.3f}， Visual similar: {:.3f} ".format(a_diveristy, v_diveristy))
        print("Audio regurize: {:.3f}， Visual regurize: {:.3f} ".format(a_re, v_re))


    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    model_name = 'best_model_of_dataset_{}_gamma_{}_pe_{}_beta{}_' \
                    'optimizer_{}_' \
                    'epoch_{}_acc_{}.pth'.format(args.dataset,
                                                args.gamma,
                                                args.pe,
                                                args.beta,
                                                args.optimizer,
                                                best_epoch, best_acc)

    save_dir = os.path.join(args.ckpt_path, model_name)

    torch.save(saved_dict, save_dir)
    print('The best model has been saved at {}.'.format(save_dir))



if __name__ == "__main__":
    main()
