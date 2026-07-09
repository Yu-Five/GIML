import copy
import csv
import os
import pickle
import librosa
from scipy import signal
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import skimage
import random
import time
from PIL import Image, ImageFilter
import pdb
import torch.nn as nn
import glob
import numpy as np
import time


class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):

        img = np.array(img)
        h, w, c = img.shape
        # np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance**2, size=(h, w, 1))
        N = np.repeat(N, c, axis=2)
        if self.variance>10:
            # img=(N/10)*255.0
            img = np.zeros_like(img)
        elif self.variance==1:
            img=img
        else:
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
        np.random.seed(0)
        N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
        if self.variance>10:
            # img=(N/10)*255.0
            img = np.zeros_like(img)
        elif self.variance==1:
            img=img
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







# 加的是高斯噪声
class KSDataset_Noise(nn.Module):
    def __init__(self, args,add_noise=False, mode='train', data_path='/root/autodl-tmp/data/kinect_sound/', val_half=False, val_modality='a'):
        super().__init__()
        self.val_half = val_half
        self.val_modality = val_modality

        f = open('./data/KineticSound/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name

        self.args = args

        label = range(len(class_list))
        data_dict = zip(class_list, label)
        data_dict = dict(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        else: 
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.video_path_list.append(video_path)
                self.data_label.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audio_path_list.append(audio_path)

        self.a_variance_list=[]
        self.v_variance_list=[]
        self.add_noise=add_noise

        if self.mode == 'train' and self.add_noise: 
            for i in range(len(self.data_label)):
                a = float(np.random.randint(low=1, high=12))
                v=float(np.random.randint(low=1, high=12))
                self.a_variance_list.append(a)
                self.v_variance_list.append(v)
            n = len(self.a_variance_list)
            half = n // 2
            indices_to_zero = random.sample(range(n), half)
            for idx in indices_to_zero:
                self.a_variance_list[idx] = 1
                self.v_variance_list[idx] = 1
        
        if self.mode == 'test' and self.add_noise:
            if self.val_half:
                for i in range(len(self.data_label)):
                    a = float(np.random.randint(low=1, high=12))
                    v=float(np.random.randint(low=1, high=12))
                    self.a_variance_list.append(a)
                    self.v_variance_list.append(v)
                n = len(self.a_variance_list)
                half = n // 2
                indices_to_zero = random.sample(range(n), half)
                for idx in indices_to_zero:
                    self.a_variance_list[idx] = 1
                    self.v_variance_list[idx] = 1
            else:
                for i in range(len(self.data_label)):
                    a = 5
                    v = 5
                    self.a_variance_list.append(a)
                    self.v_variance_list.append(v)

        elif self.mode == 'testt':
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
                for i in range(len(self.data_label)):
                    v_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'v':
                for i in range(len(self.data_label)):
                    a_variance = 11
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'av':
                # 两个模态都有
                for i in range(len(self.data_label)):
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
            else:
                raise ValueError(f"不支持的modality_test值: {self.modality_test}。必须是'a', 'v', 或'av'")

    def __len__(self):
        # return 10000

        return len(self.data_label)

    def __getitem__(self, idx):

        
        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            audio_variance = self.a_variance_list[idx]
        else:
            visual_variance=1
            audio_variance=1
        
        visual_noise_process=AddGaussianNoise(variance=visual_variance)
        audio_noise_process=AddGaussianNoise_spec(variance=audio_variance)
        
        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)


        # print(np.mean(spectrogram))

        # if self.mode=='train':
        spectrogram = audio_noise_process(spectrogram)
        # else:
        #     audio_noise_process=AddGaussianNoise_spec(variance=1)
        #     spectrogram = audio_noise_process(spectrogram)
        spectrogram=np.array(spectrogram)
        

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize(size=(224, 224)),
                visual_noise_process,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                # AddGaussianNoise(variance=1),
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

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.data_label[idx]
        # print(label)

        return spectrogram, images, label,visual_variance,audio_variance










class KSDataset_Noise_Mask(nn.Module):
    def __init__(self, args,add_noise=False, mode='train', data_path='/root/autodl-tmp/data/kinect_sound/', val_modality='a', val_half=False):
        super().__init__()
        self.add_noise = add_noise
        self.val_half = val_half
        self.val_modality = val_modality
        f = open('./data/KineticSound/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name

        self.args = args

        label = range(len(class_list))
        data_dict = zip(class_list, label)
        data_dict = dict(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        else:
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        self.data_label = []
        self.video_path_list = []
        self.audio_path_list = []

        remove_list = []  # 移除损坏视频

        # i=0
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.video_path_list.append(video_path)
                self.data_label.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audio_path_list.append(audio_path)

        self.a_variance_list=[]
        self.v_variance_list=[]
        if self.mode == 'train' and self.add_noise: 
            for i in range(len(self.data_label)):
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
        
        if self.mode == 'test' and self.add_noise:
            if self.val_half:
                for i in range(len(self.data_label)):
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
                for i in range(len(self.data_label)):
                    a = 5
                    v = 5
                    self.a_variance_list.append(a)
                    self.v_variance_list.append(v)

        if self.mode == 'testt':
            self.visual_missing_rate = args.visual_missing_rate
            self.audio_missing_rate = args.audio_missing_rate
            
            if self.val_modality == 'a':
                for i in range(len(self.data_label)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = 11  # 视觉缺失
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'v':
                for i in range(len(self.data_label)):
                    a_variance = 11  # 音频缺失
                    v_variance = self.visual_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
                
            elif self.val_modality == 'av':
                # 两个模态都有
                for i in range(len(self.data_label)):
                    a_variance = self.audio_missing_rate * 10 + 1
                    v_variance = self.visual_missing_rate * 10 + 1
                    self.a_variance_list.append(a_variance)
                    self.v_variance_list.append(v_variance)
            else:
                raise ValueError(f"不支持的modality_test值: {self.modality_test}。必须是'a', 'v', 或'av'")

    def __len__(self):
        # return 10000

        return len(self.data_label)

    def __getitem__(self, idx):

        
        if self.add_noise:
            visual_variance = self.v_variance_list[idx]
            audio_variance = self.a_variance_list[idx]
            # print(visual_variance, audio_variance)

        else:
            visual_variance=1
            audio_variance=1
        
        visual_noise_process=AddMaskNoise(variance=visual_variance)
        audio_noise_process=AddMaskNoise_spec(variance=audio_variance)
        
        # audio
        sample, rate = librosa.load(self.audio_path_list[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)


        # print(np.mean(spectrogram))

        # if self.mode=='train':
        spectrogram = audio_noise_process(spectrogram)
        # else:
        #     audio_noise_process=AddGaussianNoise_spec(variance=1)
        #     spectrogram = audio_noise_process(spectrogram)
        spectrogram=np.array(spectrogram)
        

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                visual_noise_process,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                # AddGaussianNoise(variance=1),
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

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.data_label[idx]
        # print(label)

        return spectrogram, images, label,visual_variance,audio_variance






