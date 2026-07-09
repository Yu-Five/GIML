import copy
import csv
import os
import pickle
import librosa
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F  
from torchvision import transforms
import pdb
import time
import numpy as np
import random


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        # random.seed(42)
        # np.random.seed(42)
        # 指数型影响；数值范围更广，同一个数量级难区分（尤其是高噪声，80、90、100等）。
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance**2, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        if self.variance > 10:
            # 和原始图像无关，有效地模拟模态完全缺失，不可预测的随机输入，自然地混合了不同强度的噪声
            # img = (N / 10) * 255.0
            img = np.zeros_like(img)
            # print(self.variance)
        elif self.variance == 1:
            # 完整数据
            img = img
        else:
            # 不同强度的噪声数据
            img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img



class AddGaussianNoise_spec(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        h, w = img.shape
        # random.seed(42)
        # np.random.seed(42)
        # 音频数据的数值明显小于图像，施加噪声强度更小
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
        if self.variance > 10:
            # img = (N / 10) * 255.0
            img = np.zeros_like(img)
        elif self.variance == 1:
            img = img
        else:
            img = N + img
        img[img > 255] = 255                       # 避免有值超过255而反转
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        return img



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





class CramedDataset_gaussian(Dataset):

    def __init__(self, args, mode='train', add_noise=False, val_modality='a', val_half=False):
        # import ipdb; ipdb.set_trace();
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.val_modality = val_modality
        self.val_half = val_half

        self.data_root = '/root/autodl-fs/CREMA-D-Only/CREMA-D/CREMAD_csv/'
        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_feature_path = self.args.visual_path
        self.audio_feature_path = self.args.audio_path

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')  # wav路径
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.num_frame),
                                           item[0])  # 包含多个image

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue
        
        self.a_variance_list=[]
        self.v_variance_list=[]
        self.add_noise = add_noise

        if self.mode == 'train' and add_noise: 
            for i in range(len(self.image)): 
                # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                a = float(np.random.randint(low=1, high=12))
                v = float(np.random.randint(low=1, high=12))
                
                self.a_variance_list.append(a)
                self.v_variance_list.append(v)

            n = len(self.a_variance_list)
            half = n // 2
            indices_to_zero = random.sample(range(n), half)
            for idx in indices_to_zero:
                self.a_variance_list[idx] = 1
                self.v_variance_list[idx] = 1  
        
        elif self.mode == 'valid' and add_noise:
            if self.val_half:
                for i in range(len(self.image)): 
                    # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                    a = float(np.random.randint(low=1, high=12))
                    v = float(np.random.randint(low=1, high=12))
                    
                    self.a_variance_list.append(a)
                    self.v_variance_list.append(v)
                n = len(self.a_variance_list)
                half = n // 2
                indices_to_zero = random.sample(range(n), half)
                for idx in indices_to_zero:
                    self.a_variance_list[idx] = 1
                    self.v_variance_list[idx] = 1  
            else:
                for i in range(len(self.image)): 
                    # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                    # a = float(np.random.randint(low=1, high=12))
                    # v = float(np.random.randint(low=1, high=12))
                    self.a_variance_list.append(5)
                    self.v_variance_list.append(5)

        elif self.mode == 'test':
            self.level_ranges = {
                "clean": [1, 1],
                "low": [3, 3],
                "medium": [6, 6],
                "high": [9, 9]
            }
            a_low = self.level_ranges[args.a_noise_intensity_level][0]
            a_high = self.level_ranges[args.a_noise_intensity_level][1] + 1
            v_low = self.level_ranges[args.v_noise_intensity_level][0]
            v_high = self.level_ranges[args.v_noise_intensity_level][1] + 1

            a_variance = float(np.random.randint(low=a_low, high=a_high))
            v_variance = float(np.random.randint(low=v_low, high=v_high))
            if self.val_modality == 'a':    
                for i in range(len(self.image)):
                    v_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'v':
                for i in range(len(self.image)):
                    a_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'av':
                # 两个模态都有
                for i in range(len(self.image)):
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
            else:
                raise ValueError(f"不支持的modality_test值: {self.modality_test}。必须是'a', 'v', 或'av'")
            

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            audio_variance = self.a_variance_list[idx]
        else:
            visual_variance = 1
            audio_variance = 1

        # Audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        
        audio_noise_process = AddGaussianNoise_spec(variance = audio_variance)
        spectrogram = audio_noise_process(spectrogram)

        spectrogram = np.array(spectrogram)

        # Visual
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            AddGaussianNoise(variance=visual_variance),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # image_samples = os.listdir(self.image[idx]) 
        # images = torch.zeros((self.args.num_frame, 3, 224, 224))
        # for i in range(self.args.num_frame):
        #     img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
        #     img = transform(img)
        #     images[i] = img

        # images = torch.permute(images, (1, 0, 2, 3))

        image_samples = os.listdir(self.image[idx])
        image_samples.sort()
        if len(image_samples)>1:
            select_index = np.random.choice(np.arange(1, len(image_samples)), size=self.args.num_frame, replace=False)
        else:
            select_index=[0]
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img
        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.label[idx]

        # return 音频频谱图、多帧图像张量、标签(情绪)、音视频噪声方差(用于正则项)
        return spectrogram, images, label, visual_variance, audio_variance
     






class CramedDataset_mask(Dataset):
    def __init__(self, args, mode='train', add_noise=False, val_modality='a', val_half=False):

        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        self.val_modality = val_modality
        self.val_half = val_half

        self.data_root = '/root/autodl-fs/CREMA-D-Only/CREMA-D/CREMAD_csv/'
        class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}

        self.visual_feature_path = self.args.visual_path
        self.audio_feature_path = self.args.audio_path

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')  # wav路径
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.args.num_frame),
                                           item[0])  # 包含多个image

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue
        
        self.a_variance_list=[]
        self.v_variance_list=[]
        self.add_noise = add_noise

        if self.mode == 'train' and add_noise: 
            for i in range(len(self.image)): 
                # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                a = float(np.random.randint(low=1, high=12))
                v = float(np.random.randint(low=1, high=12))
                # choices = [1, 3, 5, 7, 9, 11]
                # a = float(np.random.choice(choices))
                # v = float(np.random.choice(choices))
                
                self.a_variance_list.append(a)
                self.v_variance_list.append(v)

            n = len(self.a_variance_list)
            half = n // 2
            indices_to_zero = random.sample(range(n), half)
            for idx in indices_to_zero:
                self.a_variance_list[idx] = 1
                self.v_variance_list[idx] = 1  
        
        elif self.mode == 'valid' and add_noise:
            if self.val_half:
                for i in range(len(self.image)): 
                    # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                    a = float(np.random.randint(low=1, high=12))
                    v = float(np.random.randint(low=1, high=12))
                    
                    self.a_variance_list.append(a)
                    self.v_variance_list.append(v)
                n = len(self.a_variance_list)
                half = n // 2
                indices_to_zero = random.sample(range(n), half)
                for idx in indices_to_zero:
                    self.a_variance_list[idx] = 1
                    self.v_variance_list[idx] = 1  
            else:
                for i in range(len(self.image)): 
                    # 每个样本的一个 scalar（噪声强度的模拟），人为为每个样本生成了一个“噪声方差标签”。
                    # a = float(np.random.randint(low=1, high=12))
                    # v = float(np.random.randint(low=1, high=12))
                    self.a_variance_list.append(5)
                    self.v_variance_list.append(5)


        elif self.mode == 'test':

            self.visual_missing_rate = args.visual_missing_rate
            self.audio_missing_rate = args.audio_missing_rate
            
            if self.val_modality == 'a':
                for i in range(len(self.image)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = 11  # 视觉缺失
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'v':
                for i in range(len(self.image)):
                    a_variance = 11  # 音频缺失
                    v_variance = self.visual_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'av':
                # 两个模态都有
                for i in range(len(self.image)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = self.visual_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
            else:
                raise ValueError(f"不支持的modality_test值: {self.modality_test}。必须是'a', 'v', 或'av'")
            

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            audio_variance = self.a_variance_list[idx]
        else:
            visual_variance = 1
            audio_variance = 1

        # Audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        resamples = np.tile(samples, 3)[:22050 * 3]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        
        audio_noise_process = AddMaskNoise_spec(variance = audio_variance)
        spectrogram = audio_noise_process(spectrogram)

        spectrogram = np.array(spectrogram)

        # Visual
        # transform = transforms.Compose([
        #     transforms.Resize(size=(224, 224)),
        #     AddMaskNoise(variance=visual_variance),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        if self.mode == 'train':
            transform = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(size=(224, 224)),
                AddMaskNoise(variance=visual_variance),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                AddMaskNoise(variance=visual_variance),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # image_samples = os.listdir(self.image[idx]) 
        # images = torch.zeros((self.args.num_frame, 3, 224, 224))
        # for i in range(self.args.num_frame):
        #     img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
        #     img = transform(img)
        #     images[i] = img

        # images = torch.permute(images, (1, 0, 2, 3))

        image_samples = os.listdir(self.image[idx])
        image_samples.sort()
        if len(image_samples)>1:
            select_index = np.random.choice(np.arange(1, len(image_samples)), size=self.args.num_frame, replace=False)
        else:
            select_index=[0]
        select_index.sort()
        images = torch.zeros((self.args.num_frame, 3, 224, 224))
        for i in range(self.args.num_frame):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img
        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.label[idx]

        # return 音频频谱图、多帧图像张量、标签(情绪)、音视频噪声方差(用于正则项)
        return spectrogram, images, label, visual_variance, audio_variance
    










