"""
Docstring for dataset.MOSIDataset
MOSI的数据集来源和参考处理: https://github.com/zehuiwu/MMML/blob/main/extract_audio.py
"""

import os
import csv
import random
import librosa
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer


def listdir_nohidden(path):
    return sorted(
        [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    )



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





class AddMaskNoise(object):
    def __init__(self, variance=1.0):
        self.variance = variance
        self.mask_prob = (variance - 1.0) / 10.0

    def __call__(self, img):
        # import ipdb; ipdb.set_trace();
        img = np.array(img)
        h, w, c = img.shape
        if self.variance == 1:
            return Image.fromarray(img.astype('uint8')).convert('RGB')
        elif self.variance >= 11:
            black_img = np.zeros_like(img)
            return Image.fromarray(black_img.astype('uint8')).convert('RGB')
        else:
            mask = np.random.random((h, w)) < self.mask_prob
            result = img.copy()
            # 对于多通道图像，将掩码应用到每个通道
            for channel in range(c):
                result[:, :, channel][mask] = 0
            return Image.fromarray(result.astype('uint8')).convert('RGB')


class AddMaskNoise_spec(object):
    def __init__(self, variance=1.0):
        self.variance = variance
        self.mask_prob = (variance - 1.0) / 10.0

    def __call__(self, img):
        # import ipdb; ipdb.set_trace();
        img = np.array(img)
        
        if self.variance == 1:
            return img
        elif self.variance >= 11:
            return np.zeros_like(img)
        else:
            h, w = img.shape
            mask = np.random.random((h, w)) < self.mask_prob
            result = img.copy()
            result[mask] = 0
            return result







class MOSIDataset_mask(Dataset):
    def __init__(self, args, mode='train', val_modality='t', add_noise=False, bert_model_path='./pretrain_weights/bert-base-uncased', root_dir='/root/autodl-tmp/data/CMU-MOSI'):
        super().__init__()

        assert mode in ['train', 'valid', 'test']
        self.args = args
        self.mode = mode
        self.val_modality = val_modality
        self.add_noise = add_noise

        self.video_path_list = []
        self.audio_path_list = []
        self.text_list = []
        self.label_list = []

        label_csv = os.path.join(root_dir, 'label.csv')
        video_root = os.path.join(
            root_dir, 'Process', 'img', mode
        )

        audio_root = os.path.join(
            root_dir, 'Process', 'wav', mode
        )


        # 读取 label.csv
        with open(label_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['mode'] != mode:
                    continue

                video_id = row['video_id']
                clip_id = row['clip_id']

                video_dir = os.path.join(
                    video_root,
                    video_id,
                    clip_id
                )

                audio_path = os.path.join(
                    audio_root,
                    video_id,
                    f'{clip_id}.wav'
                )

                # 跳过缺失数据
                if not os.path.isdir(video_dir):
                    continue
                if not os.path.isfile(audio_path):
                    continue

                self.video_path_list.append(video_dir)
                self.audio_path_list.append(audio_path)
                self.text_list.append(row['text'])
                self.label_list.append(float(row['label']))

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.max_length = args.max_length
        # self.face_cascade = cv2.CascadeClassifier(
        #     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        # )

        self.t_variance_list=[]
        self.v_variance_list=[]
        self.a_variance_list=[]

        if self.mode == 'train' and self.add_noise:
            for i in range(len(self.video_path_list)):
                # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                t = float(np.random.randint(low=1, high=12))
                # choices = [1, 2, 3, 4, 5, 6, 11]
                # t = float(np.random.choice(choices))
                v = float(np.random.randint(low=1, high=12))
                a = float(np.random.randint(low=1, high=12))
                
                self.t_variance_list.append(t)
                self.v_variance_list.append(v)
                self.a_variance_list.append(a)

            n = len(self.t_variance_list)
            half = n // 2
            indices_to_zero = random.sample(range(n), half)
            for idx in indices_to_zero:
                self.t_variance_list[idx] = 1
                self.v_variance_list[idx] = 1
                self.a_variance_list[idx] = 1
            
        
        elif self.mode == 'valid' and self.add_noise: 
            for i in range(len(self.video_path_list)):
                # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                self.t_variance_list.append(5)
                self.v_variance_list.append(5)
                self.a_variance_list.append(5)

        elif self.mode == 'test':
            self.visual_missing_rate = args.visual_missing_rate
            self.audio_missing_rate = args.audio_missing_rate
            self.text_missing_rate = args.text_missing_rate

            self.a_variance_list = []      # 音频缺失程度
            self.v_variance_list = []      # 视觉缺失程度
            self.t_variance_list = []    # 文本缺失程度

            if self.val_modality == 'a':
                for i in range(len(self.video_path_list)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = 11
                    t_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            elif self.val_modality == 'v':
                for i in range(len(self.video_path_list)):
                    a_variance = 11
                    v_variance = self.visual_missing_rate * 10 + 1
                    t_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            elif self.val_modality == 't':          # 只有文本
                for i in range(len(self.video_path_list)):
                    a_variance = 11
                    v_variance = 11
                    t_variance = self.text_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            elif self.val_modality == 'av':
                for i in range(len(self.video_path_list)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = self.visual_missing_rate * 10 + 1
                    t_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            elif self.val_modality == 'at':         # 音频+文本
                for i in range(len(self.video_path_list)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = 11
                    t_variance = self.text_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            elif self.val_modality == 'vt':         # 视觉+文本
                for i in range(len(self.video_path_list)):
                    a_variance = 11
                    v_variance = self.visual_missing_rate * 10 + 1
                    t_variance = self.text_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            elif self.val_modality == 'avt':        # 全部存在
                for i in range(len(self.video_path_list)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = self.visual_missing_rate * 10 + 1
                    t_variance = self.text_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                    self.t_variance_list.append(t_variance)
            else:
                raise ValueError(f"Unknown val_modality: {self.val_modality}")


    def __len__(self):
        return len(self.label_list)

    
    def __getitem__(self, idx):
        
        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            text_variance = self.t_variance_list[idx]
            audio_variance = self.a_variance_list[idx]

        else:
            visual_variance = 1
            text_variance = 1
            audio_variance = 1

        # text
        text = self.text_list[idx]
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
    

        visual_noise_process=AddMaskNoise(variance=visual_variance)
        audio_noise_process=AddMaskNoise_spec(variance=audio_variance)
        
        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)   

        # while len(sample) / rate < 5.:
        #     sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        # start_point = random.randint(a=0, b=(len(sample) - rate * 5))
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        
        # if self.mode=='train':
        spectrogram = audio_noise_process(spectrogram)
        # else:
        #     audio_noise_process=AddGaussianNoise_spec(variance=1)
        #     spectrogram = audio_noise_process(spectrogram)
        spectrogram=np.array(spectrogram)
        

        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                transforms.Resize(size=(224, 224)),
                # transforms.RandomHorizontalFlip(),
                visual_noise_process,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else: 
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                visual_noise_process, 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = listdir_nohidden(self.video_path_list[idx])
        # print(len(image_samples))
        select_index = np.random.choice(len(image_samples), size=self.args.num_frame, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            try:
                img = Image.open(image_samples[select_index[i]]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue

            img = transform(img)
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))


        # label
        raw_label = self.label_list[idx]

        if raw_label < 0: 
            label = 0
        else: 
            label = 1

        label = torch.tensor(label, dtype=torch.long)
        # print(label)

        return spectrogram, images, text_data, label, audio_variance, visual_variance, text_variance



