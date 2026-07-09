"""
Docstring for OGM_PE_ARL.dataset.MVSADataset
直接可以使用的
https://github.com/QingyangZhang/QMF
"""

import os
import json
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


class AddMaskNoise_Image(object):
    '''
    图像掩码噪声添加器
    variance: 1.0表示无缺失，11.0表示完全缺失
    '''
    def __init__(self, variance=1.0):
        self.variance = variance
        self.mask_prob = (variance - 1.0) / 10.0
    
    def __call__(self, img):
        if self.variance <= 1:
            return img
        
        if self.variance >= 11:
            img_array = np.array(img)
            black_img = np.zeros_like(img_array)
            return Image.fromarray(black_img.astype('uint8')).convert('RGB')
        
        # 部分缺失
        img_array = np.array(img)
        h, w, c = img_array.shape
        
        # 创建随机mask
        mask = np.random.random((h, w)) < self.mask_prob
        mask_3d = np.stack([mask]*c, axis=-1)
        
        # 应用mask
        result = img_array.copy()
        result[mask_3d] = 0
        
        return Image.fromarray(result.astype('uint8')).convert('RGB')


class AddMaskNoise_Text(object):
    '''
    文本缺失模拟器（word-level，真实缺失）
    variance: 1.0 表示无缺失，11.0 表示完全缺失
    '''
    def __init__(self, variance=1.0):
        self.variance = variance
        self.missing_rate = (variance - 1.0) / 10.0

    def __call__(self, text):
        if self.missing_rate == 0:
            return text

        if self.missing_rate == 1:
            # 如果完全缺失，返回一个和原文长度相同的 `[MASK]` 串
            return " ".join(["[MASK]"] * len(text.split()))

        words = text.split()
        if not words:
            print(f"words: {words}")
            return text

        n_words = len(words)
        n_drop = int(round(n_words * self.missing_rate))  # 计算需要掩盖的词数

        if n_drop <= 0:
            # print(f"n_drop: {n_drop}")
            return text
        
        drop_indices = set(random.sample(range(n_words), min(n_drop, n_words)))
        masked_words = [
            '[MASK]' if i in drop_indices else w
            for i, w in enumerate(words)
        ]

        return " ".join(masked_words)



class MVSADataset_mask(Dataset):
    def __init__(self, args, mode='train', val_modality='t', add_noise=False, bert_model_path='./pretrain_weights/bert-base-uncased'):
        self.args = args
        self.text = []
        self.image = []
        self.label = []
        self.mode = mode
        self.val_modality = val_modality
        self.add_noise = add_noise

        self.data_root = "/root/autodl-tmp/data/MVSA_Single/"
        self.train_jsonl = os.path.join(self.data_root, 'train.jsonl')
        self.val_jsonl = os.path.join(self.data_root, 'val.jsonl')
        self.test_jsonl = os.path.join(self.data_root, 'test.jsonl')

        if mode == 'train':
            jsonl_file = self.train_jsonl
        elif mode == 'valid':
            jsonl_file = self.val_jsonl
        elif mode == 'test':
            jsonl_file = self.test_jsonl
        else:
            raise ValueError(f"Invalid mode: {mode}")

        class_dict = {'neutral': 0, 'positive': 1, 'negative': 2}
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.text.append(data['text'])
                self.image.append(os.path.join(self.data_root, data['img']))
                self.label.append(class_dict[data['label']])

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.max_length = args.max_length

        self.t_variance_list=[]
        self.v_variance_list=[]

        if self.mode == 'train' and self.add_noise:
            for i in range(len(self.image)):
                # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                t = float(np.random.randint(low=1, high=12))
                # choices = [1, 2, 3, 4, 5, 6, 11]
                # t = float(np.random.choice(choices))
                v = float(np.random.randint(low=1, high=12))
                
                self.t_variance_list.append(t)
                self.v_variance_list.append(v)

            n = len(self.t_variance_list)
            half = n // 2
            indices_to_zero = random.sample(range(n), half)
            for idx in indices_to_zero:
                self.t_variance_list[idx] = 1
                self.v_variance_list[idx] = 1
            
        
        elif self.mode == 'valid' and self.add_noise: 
            for i in range(len(self.image)):
                # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                self.t_variance_list.append(5)
                self.v_variance_list.append(5)


        elif self.mode == 'test':
            self.visual_missing_rate = args.visual_missing_rate
            self.text_missing_rate = args.text_missing_rate
            
            if self.val_modality == 't':
                for i in range(len(self.label)):
                    t_variance = self.text_missing_rate * 10 + 1
                    v_variance = 11  # 视觉缺失
                    self.t_variance_list.append(t_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'v':
                for i in range(len(self.label)):
                    t_variance = 11  # 音频缺失
                    v_variance = self.visual_missing_rate * 10 + 1
                    self.t_variance_list.append(t_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'vt':
                # 两个模态都有
                for i in range(len(self.label)):
                    t_variance = self.text_missing_rate * 10 + 1
                    v_variance = self.visual_missing_rate * 10 + 1
                    self.t_variance_list.append(t_variance)
                    self.v_variance_list.append(v_variance)


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # import ipdb; ipdb.set_trace();
        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            text_variance = self.t_variance_list[idx]
        else:
            visual_variance = 1
            text_variance = 1

        text = self.text[idx]
        text_noise_process = AddMaskNoise_Text(variance=text_variance)
        noisy_text = text_noise_process(text) 

        encoding = self.tokenizer(
            noisy_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # squeeze batch 维度
        text_data = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
    


        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                AddMaskNoise_Image(variance=visual_variance),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else: 
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                AddMaskNoise_Image(variance=visual_variance), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


        img = Image.open(self.image[idx]).convert('RGB')
        images = self.transform(img)
        # 相当于num_frame，为了和AT保持一致
        images = images.unsqueeze(1)
        # label
        label = self.label[idx]
        
        return text_data, images, label, text_variance, visual_variance





