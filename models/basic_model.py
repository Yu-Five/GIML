import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18
from .bert import Bert_Model
from .fusion_modules import SumFusion, ConcatFusion, FiLM, GatedFusion, ConcatFusion_Swin, ConcatFusion_AUXI, \
    GatedFusion_AUXI, SumFusion_AUXI, FiLM_AUXI, ShareWeightFusion_AUXI, ConcatFusion_AUXI_3
import numpy as np
# from models.swin_transformer import SwinTransformer


def modality_drop(x_rgb, x_depth, p, args=None):
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]

    if p == [0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array((1 / 3, 1 / 3, 1 / 3))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        # if [0, 1] not in p:
        #     p[0] = [0, 1]
        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 2)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]

    if x_rgb.shape[0] != x_depth.shape[0]:
        pv = torch.repeat_interleave(p, args.use_video_frames, dim=0)
        # print(pv.shape)
        x_depth = x_depth * pv[:, 1]
    else:
        x_depth = x_depth * p[:, 1]

    return x_rgb, x_depth, p


class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden)
        self.w_2 = nn.Linear(d_hidden, n_head)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    #     self.init_weights()
    #
    # def init_weights(self):
    #     nn.init.xavier_uniform_(self.w_1.weight)
    #     nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PCME(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return out


class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        elif args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        elif args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        self.unimodal_fc = nn.Linear(512, n_classes)

    def forward(self, audio, visual):

        if self.modality == 'full':
            if self.pe:
                # print("here!!!!!!!!!!!!")
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # v=v*0

                a_out = self.unimodal_fc(a)
                v_out = self.unimodal_fc(v)

                a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                a_out = self.unimodal_fc(a)
                v_out = self.unimodal_fc(v)

                a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:

                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0
        else:
            return 0, 0, 0



        
class AVClassifier_AUXI_AV(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_AV, self).__init__()

        if args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6

        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)


        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
           
        self.pe = args.pe
        self.args = args


        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual):

            a, a_mul, a_std = self.audio_net(audio)  # only feature
            v, v_mul, v_std = self.visual_net(visual)

            if self.args.drop: 
                a, v, p = modality_drop(a, v, self.args.p, args=self.args)

            a_feature = a
            v_feature = v

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            v_std_in = v_std.view(B, -1, C, H, W)
            v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

            a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

            a_std_in = torch.flatten(a_std_in, 1)
            v_std_in = torch.flatten(v_std_in, 1)

            v_std_in = v_std.view(B, -1, C, H, W)
            v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

            a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

            a_std_in = torch.flatten(a_std_in, 1)
            v_std_in = torch.flatten(v_std_in, 1)

            a_std_fc=self.audio_variance_estimator(a_std_in.detach())
            v_std_fc=self.visual_variance_estimator(v_std_in.detach())

        

            a_std_fc = (a_std_fc * 0.5).exp()
            v_std_fc = (v_std_fc * 0.5).exp()

            if self.training:
                if self.args.current_epoch<self.args.cylcle_epoch+10:
                    weight_a,weight_v=1,1
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # weight_a,weight_v=1,1
                    # teaching force
                    # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                    # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                    # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
            else:
                weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # weight_a,weight_v=1,1
            # # a_out=self.unimodal_fc(a)
            # # v_out=self.unimodal_fc(v)
            # weight_a,weight_v=1,1
            # print(weight_a,weight_v)
            # print(weight_a, weight_v)
            a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

            return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc
        




        
class AVClassifier_AUXI_AV_NoDecouple(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_AV_NoDecouple, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'KineticSound':
            n_classes = 34

        elif args.dataset == 'CREMAD':
            n_classes = 6
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes)
        

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)
            

        self.pe = args.pe
        self.args = args
        
        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual):


        if self.pe:
            a, a_mul, a_std = self.audio_net(audio)  # only feature
            v, v_mul, v_std = self.visual_net(visual)

            a = a_mul
            v = v_mul

            if self.args.drop:
                a, v, p = modality_drop(a, v, self.args.p, args=self.args)

            a_feature = a
            v_feature = v

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            v_std_in = v_std.view(B, -1, C, H, W)
            v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

            a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

            a_std_in = torch.flatten(a_std_in, 1)
            v_std_in = torch.flatten(v_std_in, 1)

            v_std_in = v_std.view(B, -1, C, H, W)
            v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

            a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

            a_std_in = torch.flatten(a_std_in, 1)
            v_std_in = torch.flatten(v_std_in, 1)

            a_std_fc=self.audio_variance_estimator(a_std_in.detach())
            v_std_fc=self.visual_variance_estimator(v_std_in.detach())

            
            
#                 a_std_fc=self.audio_variance_estimator(a.detach())
#                 v_std_fc=self.visual_variance_estimator(v.detach())

            a_std_fc = (a_std_fc * 0.5).exp()
            v_std_fc = (v_std_fc * 0.5).exp()

            # if self.training:
            #     if self.args.current_epoch<self.args.cylcle_epoch+10:
            #         weight_a,weight_v=1,1
            #     else:
            #         weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
            #     # weight_a,weight_v=1,1
            #         # teaching force
            #         # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
            #         # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
            #         # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
            # else:
            #     weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # weight_a,weight_v=1,1
            # # a_out=self.unimodal_fc(a)
            # # v_out=self.unimodal_fc(v)
            # weight_a,weight_v=1,1
            # print(weight_a,weight_v)
            weight_a,weight_v=1,1
            a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

            return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc
            


  
class AVClassifier_AUXI_TV(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_TV, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'MVSA_Single':
            n_classes = 3
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=(512+512))
    
        self.text_net = Bert_Model(args=args)
        self.visual_net = resnet18(modality='visual', args=args)

        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.text_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, text, visual):
        # import ipdb; ipdb.set_trace();

        if self.pe:
            t, t_mul, t_std = self.text_net(text)  # only feature
            v, v_mul, v_std = self.visual_net(visual)

            if self.args.drop:
                t, v, p = modality_drop(t, v, self.args.p, args=self.args)

            t_feature = t
            v_feature = v

            (_, C, H, W) = v.size()
            B = t.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)
            v = F.adaptive_avg_pool3d(v, 1)
            v = torch.flatten(v, 1)
            v_std_in = v_std.view(B, -1, C, H, W)
            v_std_in = v_std_in.permute(0, 2, 1, 3, 4)
            v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)
            v_std_in = torch.flatten(v_std_in, 1)

            t_std_in = t_std
            
            t_std_fc=self.text_variance_estimator(t_std_in.detach())
            v_std_fc=self.visual_variance_estimator(v_std_in.detach())

            
            
#                 a_std_fc=self.audio_variance_estimator(a.detach())
#                 v_std_fc=self.visual_variance_estimator(v.detach())

            t_std_fc = (t_std_fc * 0.5).exp()
            v_std_fc = (v_std_fc * 0.5).exp()

            if self.training:
                if self.args.current_epoch<self.args.cylcle_epoch+10:
                    weight_a,weight_v=1,1
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+t_std_fc**2),2*t_std_fc**2/(v_std_fc**2+t_std_fc**2)

                    # teaching force
                    # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                    # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                    # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
            else:
                weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+t_std_fc**2),2*t_std_fc**2/(v_std_fc**2+t_std_fc**2)
            # # a_out=self.unimodal_fc(a)
            # # v_out=self.unimodal_fc(v)
            # weight_a,weight_v=1,1
            # print(weight_a,weight_v)
            
            t_out, v_out, out = self.fusion_module(t+weight_a*t, v+weight_v*v)  # av 是原来的，out是融合结果

            return t, v, out, t_feature, v_feature, t_mul, t_std, v_mul, v_std, t_out, v_out,t_std_fc,v_std_fc
            




class AVClassifier_AUXI_ATV(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_ATV, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'MOSI':
            n_classes = 2

        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.fusion_module = ConcatFusion_AUXI_3(output_dim=n_classes, input_dim=(512+512+512))

        self.text_net = Bert_Model(args=args)
        self.visual_net = resnet18(modality='visual', args=args)
        self.audio_net = resnet18(modality='audio', args=args)

        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.text_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual, text):
        # import ipdb; ipdb.set_trace();

        if self.pe:
            t, t_mul, t_std = self.text_net(text)  # only feature
            v, v_mul, v_std = self.visual_net(visual)
            a, a_mul, a_std = self.audio_net(audio)

            if self.args.drop:
                a, v, p = modality_drop(a, v, self.args.p, args=self.args)

            a_feature = a
            v_feature = v
            t_feature = t

            (_, C, H, W) = v.size()
            B = a.size()[0]
            v = v.view(B, -1, C, H, W)
            v = v.permute(0, 2, 1, 3, 4)

            a = F.adaptive_avg_pool2d(a, 1)
            v = F.adaptive_avg_pool3d(v, 1)

            a = torch.flatten(a, 1)
            v = torch.flatten(v, 1)

            v_std_in = v_std.view(B, -1, C, H, W)
            v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

            a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

            a_std_in = torch.flatten(a_std_in, 1)
            v_std_in = torch.flatten(v_std_in, 1)


            # (_, C, H, W) = v.size()
            # B = a.size()[0]
            # v = v.view(B, -1, C, H, W)
            # # v = v.permute(0, 2, 1, 3, 4)
            # T = v.size(1)
            # v = v.view(B * T, C, H, W)

            # a = F.adaptive_avg_pool2d(a, 1)
            # # v = F.adaptive_avg_pool3d(v, 1)
            # v = F.adaptive_avg_pool2d(v, 1)

            # a = torch.flatten(a, 1)
            # v = torch.flatten(v, 1)

            # v = v.view(B, T, C)
            # v, _ = self.visual_lstm_mul(v)
            # # v = torch.mean(v, dim=1)
            # v = v[:, -1, :]

            # v_std_in = v_std.view(B, -1, C, H, W)
            # # v_std_in = v_std_in.permute(0, 2, 1, 3, 4)
            # v_std_in = v_std_in.view(B * T, C, H, W)

            # a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            # # v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)
            # v_std_in = F.adaptive_avg_pool2d(v_std_in, 1)

            # a_std_in = torch.flatten(a_std_in, 1)
            # v_std_in = torch.flatten(v_std_in, 1)

            # v_std_in = v_std_in.view(B, T, -1)
            # v_std_in, _ = self.visual_lstm_std(v_std_in)
            # # v_std_in = torch.mean(v_std_in, dim=1)
            # v_std_in = v_std_in[:, -1, :]


            t_std_in = t_std
            
            t_std_fc=self.text_variance_estimator(t_std_in.detach())
            v_std_fc=self.visual_variance_estimator(v_std_in.detach())
            a_std_fc=self.audio_variance_estimator(a_std_in.detach())
            
            
#                 a_std_fc=self.audio_variance_estimator(a.detach())
#                 v_std_fc=self.visual_variance_estimator(v.detach())

            t_std_fc = (t_std_fc * 0.5).exp()
            v_std_fc = (v_std_fc * 0.5).exp()
            a_std_fc = (a_std_fc * 0.5).exp()

            if self.training:
                if self.args.current_epoch<self.args.cylcle_epoch+10:
                    weight_a, weight_v, weight_t = 1, 1, 1
                else:
                    weight_a, weight_v, weight_t = 3*(v_std_fc**2+t_std_fc**2)/(v_std_fc**2+t_std_fc**2+a_std_fc**2), 3*(t_std_fc**2+a_std_fc**2)/(v_std_fc**2+t_std_fc**2+a_std_fc**2), 3*(v_std_fc**2+a_std_fc**2)/(v_std_fc**2+t_std_fc**2+a_std_fc**2)

                    # teaching force
                    # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                    # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                    # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
            else:
                weight_a, weight_v, weight_t = 3*(v_std_fc**2+t_std_fc**2)/(v_std_fc**2+t_std_fc**2+a_std_fc**2), 3*(t_std_fc**2+a_std_fc**2)/(v_std_fc**2+t_std_fc**2+a_std_fc**2), 3*(v_std_fc**2+a_std_fc**2)/(v_std_fc**2+t_std_fc**2+a_std_fc**2)
            # # a_out=self.unimodal_fc(a)
            # # v_out=self.unimodal_fc(v)
            # weight_a,weight_v=1,1
            # print(weight_a,weight_v)
            
            a_out, v_out, t_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v, t+weight_t*t)  # av 是原来的，out是融合结果

            return a, v, t, out, a_feature, v_feature, t_feature, a_mul, a_std, v_mul, v_std, t_mul, t_std, a_out, v_out, t_out, a_std_fc, v_std_fc, t_std_fc
        
            # a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc

            





class AVClassifier_AUXI_AT(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_AT, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'MOSI':
            n_classes = 2

        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        
        self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=(512+512))


        self.text_net = Bert_Model(args=args)
        self.audio_net = resnet18(modality='audio', args=args)

        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.text_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, text):
        # import ipdb; ipdb.set_trace();

        if self.pe:
            t, t_mul, t_std = self.text_net(text)  # only feature
            a, a_mul, a_std = self.audio_net(audio)

            if self.args.drop:
                a, v, p = modality_drop(a, v, self.args.p, args=self.args)

            a_feature = a
            t_feature = t

            B = a.size()[0]
            a = F.adaptive_avg_pool2d(a, 1)
            a = torch.flatten(a, 1)
            a_std_in = F.adaptive_avg_pool2d(a_std, 1)
            a_std_in = torch.flatten(a_std_in, 1)
            t_std_in = t_std
            t_std_fc=self.text_variance_estimator(t_std_in.detach())
            a_std_fc=self.audio_variance_estimator(a_std_in.detach())
            
#                 a_std_fc=self.audio_variance_estimator(a.detach())
#                 v_std_fc=self.visual_variance_estimator(v.detach())

            t_std_fc = (t_std_fc * 0.5).exp()
            a_std_fc = (a_std_fc * 0.5).exp()

            if self.training:
                if self.args.current_epoch<self.args.cylcle_epoch+10:
                    weight_a, weight_t = 1, 1
                else:
                    weight_a, weight_t = 2*(t_std_fc**2)/(t_std_fc**2+a_std_fc**2), 2*(a_std_fc**2)/(t_std_fc**2+a_std_fc**2)

                    # teaching force
                    # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                    # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                    # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
            else:
                weight_a, weight_t = 2*(t_std_fc**2)/(t_std_fc**2+a_std_fc**2), 2*(a_std_fc**2)/(t_std_fc**2+a_std_fc**2)
            # # a_out=self.unimodal_fc(a)
            # # v_out=self.unimodal_fc(v)
            # weight_a,weight_v=1,1
            # print(weight_a,weight_v)
            
            a_out, t_out, out = self.fusion_module(a+weight_a*a, t+weight_t*t)  # av 是原来的，out是融合结果

            return a, t, out, a_feature, t_feature, a_mul, a_std, t_mul, t_std, a_out, t_out, a_std_fc, t_std_fc
        
            # a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc

    



       

class AVClassifier_AUXI_RD(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_RD, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'NVGesture':
            n_classes = 25
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=(512+512))
        

        self.rgb_net = resnet18(modality='visual', args=args)
        self.depth_net = resnet18(modality='depth', args=args)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.rgb_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.depth_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))

    def forward(self, rgb, depth):
        # import ipdb; ipdb.set_trace();
        if self.pe:
            rgb, rgb_mul, rgb_std = self.rgb_net(rgb)
            depth, depth_mul, depth_std = self.depth_net(depth)

        # if self.args.drop:
        #     a, rgb, p = modality_drop(a, rgb, self.args.p, args=self.args)

        rgb_feature = rgb
        depth_feature = depth

        # RGB
        (_, C, H, W) = rgb.size()
        # B = self.args.batch_size  # 这样不行，因为最后有不足batch_size的
        B = rgb.shape[0] // self.args.num_frame
        # import ipdb; ipdb.set_trace();
        rgb = rgb.view(B, -1, C, H, W)
        rgb = rgb.permute(0, 2, 1, 3, 4)
        rgb = F.adaptive_avg_pool3d(rgb, 1)
        rgb = torch.flatten(rgb, 1)

        rgb_std_in = rgb_std.view(B, -1, C, H, W)
        rgb_std_in = rgb_std_in.permute(0, 2, 1, 3, 4)
        rgb_std_in = F.adaptive_avg_pool3d(rgb_std_in, 1)
        rgb_std_in = torch.flatten(rgb_std_in, 1)

        # Depth
        (_, C, H, W) = depth.size()
        B = depth.shape[0] // self.args.num_frame
        depth = depth.view(B, -1, C, H, W)
        depth = depth.permute(0, 2, 1, 3, 4)
        depth = F.adaptive_avg_pool3d(depth, 1)
        depth = torch.flatten(depth, 1)

        depth_std_in = depth_std.view(B, -1, C, H, W)
        depth_std_in = depth_std_in.permute(0, 2, 1, 3, 4)
        depth_std_in = F.adaptive_avg_pool3d(depth_std_in, 1)
        depth_std_in = torch.flatten(depth_std_in, 1)


        rgb_std_fc=self.rgb_variance_estimator(rgb_std_in.detach())
        depth_std_fc=self.depth_variance_estimator(depth_std_in.detach())
        # a_std_fc=self.audio_variance_estimator(a.detach())
        # v_std_fc=self.visual_variance_estimator(v.detach())
        rgb_std_fc = (rgb_std_fc * 0.5).exp()
        depth_std_fc = (depth_std_fc * 0.5).exp()

        if self.training:
            if self.args.current_epoch<self.args.cylcle_epoch+10:
                weight_rgb, weight_depth = 1, 1
            else:
                weight_rgb, weight_depth = 2*(depth_std_fc**2)/(rgb_std_fc**2+depth_std_fc**2), 2*(rgb_std_fc**2)/(rgb_std_fc**2+depth_std_fc**2)
            # teaching force
            # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
            # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
            # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
        else:
            weight_rgb, weight_depth = 2*(depth_std_fc**2)/(rgb_std_fc**2+depth_std_fc**2), 2*(rgb_std_fc**2)/(rgb_std_fc**2+depth_std_fc**2)

        # # v_out=self.unimodal_fc(v)
        # weight_a,weight_v=1,1
        # print(weight_a,weight_v)

        rgb_out, depth_out, out = self.fusion_module(rgb+weight_rgb*rgb, depth+weight_depth*depth)

        return rgb, depth, out, rgb_feature, depth_feature, rgb_mul, rgb_std, depth_mul, depth_std, rgb_out, depth_out,rgb_std_fc,depth_std_fc
        # a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc




class AVClassifier_PE(nn.Module):
    def __init__(self, args):
        super(AVClassifier_PE, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        self.audio_net = resnet18(modality='audio', args=args)
        self.visual_net = resnet18(modality='visual', args=args)

    def forward(self, audio, visual):

        a = self.audio_net(audio)  # only feature
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

        return a, v, out




     
class AVClassifier_AUXI_AVT_Unimodel(nn.Module):
    def __init__(self, args):
        super(AVClassifier_AUXI_AVT_Unimodel, self).__init__()

        fusion = args.fusion_method
        if args.dataset == 'MOSI':
            n_classes = 2
        elif args.dataset == 'KineticSound':
            n_classes = 34
        elif args.dataset == 'kinect400':
            n_classes = 400
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

        if fusion == 'sum':
            self.fusion_module = SumFusion_AUXI(output_dim=n_classes)
        elif fusion == 'concat':
            if args.dataset == 'kinect400':
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
            else:
                self.fusion_module = ConcatFusion_AUXI(output_dim=n_classes, input_dim=1024)
        elif fusion == 'film':
            self.fusion_module = FiLM_AUXI(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion_AUXI(output_dim=n_classes, x_gate=True)
        elif fusion == 'share':
            self.fusion_module = ShareWeightFusion_AUXI(output_dim=n_classes)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))

        if args.modality == 'full':
            self.audio_net = resnet18(modality='audio', args=args)
            self.visual_net = resnet18(modality='visual', args=args)

        if args.modality == 'visual':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
            else:
                self.visual_net = resnet18(modality='visual', args=args)
                self.visual_classifier = nn.Linear(512, n_classes)
        if args.modality == 'audio':
            if args.dataset == 'kinect400':
                print("resnet50")
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
            else:
                self.audio_net = resnet18(modality='audio', args=args)
                self.audio_classifier = nn.Linear(512, n_classes)
        self.pe = args.pe
        self.modality = args.modality
        self.args = args

        # self.audio_mu = nn.Linear(512, 512)
        # self.audio_logval = PCME(512, 512, 256)
        # self.visual_mu = nn.Linear(512, 512)
        # self.visual_logval = PCME(512, 512, 256)

        self.visual_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))
        self.audio_variance_estimator=nn.Sequential(nn.Linear(512,256),nn.Dropout(0.1),nn.Linear(256,1))


    def forward(self, audio, visual):

        if self.modality == 'full':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature
                v, v_mul, v_std = self.visual_net(visual)

                if self.args.drop:
                    a, v, p = modality_drop(a, v, self.args.p, args=self.args)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                v_std_in = v_std.view(B, -1, C, H, W)
                v_std_in = v_std_in.permute(0, 2, 1, 3, 4)

                a_std_in = F.adaptive_avg_pool2d(a_std, 1)
                v_std_in = F.adaptive_avg_pool3d(v_std_in, 1)

                a_std_in = torch.flatten(a_std_in, 1)
                v_std_in = torch.flatten(v_std_in, 1)
                

                a_std_fc=self.audio_variance_estimator(a_std_in.detach())
                v_std_fc=self.visual_variance_estimator(v_std_in.detach())

                
                
#                 a_std_fc=self.audio_variance_estimator(a.detach())
#                 v_std_fc=self.visual_variance_estimator(v.detach())

                a_std_fc = (a_std_fc * 0.5).exp()
                v_std_fc = (v_std_fc * 0.5).exp()

                if self.training:
                    if self.args.current_epoch<self.args.cylcle_epoch+10:
                        weight_a,weight_v=1,1
                    else:
                        weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)

                        # teaching force
                        # a_varinace_label=torch.unsqueeze(a_varinace_label.float(),dim=1).cuda()
                        # v_varinace_label=torch.unsqueeze(v_varinace_label.float(),dim=1).cuda()
                        # weight_a,weight_v=2*v_varinace_label**2/(v_varinace_label**2+a_varinace_label**2),2*a_varinace_label**2/(v_varinace_label**2+a_varinace_label**2)
                else:
                    weight_a,weight_v=2*v_std_fc**2/(v_std_fc**2+a_std_fc**2),2*a_std_fc**2/(v_std_fc**2+a_std_fc**2)
                # # a_out=self.unimodal_fc(a)
                # # v_out=self.unimodal_fc(v)
                # weight_a,weight_v=1,1
                # print(weight_a,weight_v)
                a_out, v_out, out = self.fusion_module(a+weight_a*a, v+weight_v*v)  # av 是原来的，out是融合结果

                return a, v, out, a_feature, v_feature, a_mul, a_std, v_mul, v_std, a_out, v_out,a_std_fc,v_std_fc
            else:
                a = self.audio_net(audio)  # only feature
                v = self.visual_net(visual)

                a_feature = a
                v_feature = v

                (_, C, H, W) = v.size()
                B = a.size()[0]
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                a = F.adaptive_avg_pool2d(a, 1)
                v = F.adaptive_avg_pool3d(v, 1)

                a = torch.flatten(a, 1)
                v = torch.flatten(v, 1)

                # a_out=self.unimodal_fc(a)
                # v_out=self.unimodal_fc(v)

                a_out, v_out, out = self.fusion_module(a, v)  # av 是原来的，out是融合结果

                # return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a, v
                return a, v, out, a_feature, v_feature, 0, 0, 0, 0, a_out, v_out
            
        elif self.modality == 'visual':

            if self.pe:
                v, v_mul, v_std = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                B = self.args.batch_size
                v = v.view(B, -1, C, H, W)
                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                return a, v, out, v_feature, v_feature, 0, 0, v_mul, v_std, 0, 0


            else:

                v = self.visual_net(visual)

                v_feature = v

                (_, C, H, W) = v.size()
                # import ipdb; ipdb.set_trace();
                B = v.shape[0] // self.args.num_frame
                v = v.view(B, -1, C, H, W)

                v = v.permute(0, 2, 1, 3, 4)

                v = F.adaptive_avg_pool3d(v, 1)

                v = torch.flatten(v, 1)

                out = self.visual_classifier(v)

                a = torch.zeros_like(v)

                # print(11111111111111)

                return a, v, out, v_feature, v_feature, 0, 0, 0, 0, 0, 0

        elif self.modality == 'audio':

            if self.pe:
                a, a_mul, a_std = self.audio_net(audio)  # only feature

                a_feature = a
                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)
                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, a_mul, a_std, 0, 0

            else:
                # print("Here!")
                a = self.audio_net(audio)  # only feature
                a_feature = a

                a = F.adaptive_avg_pool2d(a, 1)

                a = torch.flatten(a, 1)

                out = self.audio_classifier(a)
                v = torch.zeros_like(a)

                return a, v, out, a_feature, a_feature, 0, 0, 0, 0, 0, 0
        else:
            return 0, 0, 0


