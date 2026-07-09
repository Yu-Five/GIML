"""
Docstring for dataset.NVGestureDataset
参考：https://github.com/ahmetgunduz/Real-time-GesRec
"""
import torch
import torch.utils.data as data
import os
import json
import random
import numpy as np
from PIL import Image
import functools
from torchvision import transforms

# ==================== 辅助函数 ====================
def pil_loader(path, modality='RGB'):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Depth':
                return img.convert('L')
            else:
                return img.convert('RGB')

def default_video_loader(frame_paths, modality='RGB'):
    frames = []
    for path in frame_paths:
        frames.append(pil_loader(path, modality))
    return frames

def get_default_video_loader():
    return functools.partial(default_video_loader, modality='RGB')

def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)

def get_class_labels(data):
    return {label: i for i, label in enumerate(data['labels'])}

def get_video_names_and_annotations(data, subset):
    video_names, annotations = [], []
    for key, value in data['database'].items():
        if value['subset'] == subset:
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])
    return video_names, annotations

def make_dataset(root_path, annotation_path, subset):
    """构建样本列表，每个样本包含视频路径、起止帧、标签等信息"""
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)

    dataset = []
    for i, video_name in enumerate(video_names):
        video_path = os.path.join(root_path, video_name)
        if not os.path.exists(video_path):
            continue

        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': i,
            'label': class_to_idx[annotations[i]['label']]
        }
        # 生成所有帧索引（后续由 temporal_transform 采样）
        sample['frame_indices'] = list(range(begin_t, end_t + 1))
        dataset.append(sample)
    return dataset





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





class NV_Noise_Mask(data.Dataset):
    def __init__(self,
                 args,
                 add_noise=False,
                 mode='train',                # 'train', 'test', 'testt'
                 val_modality='RGB',         
                 val_half=False,            
                 video_loader=get_default_video_loader()):     
        super().__init__()
        self.args = args
        self.add_noise = add_noise
        self.mode = mode
        self.val_modality = val_modality
        self.val_half = val_half
        # import ipdb; ipdb.set_trace();
        self.video_loader = video_loader

        # import ipdb; ipdb.set_trace();

        if self.mode == 'train':
            subset = "training"
        else:
            subset = "validation"
        annotation_path = "/root/autodl-tmp/data/NVGesture/data/annotation_nvGesture/nvall_but_None.json"
        root_path = "/root/autodl-tmp/data/NVGesture/data/"
        self.data = make_dataset(root_path, annotation_path, subset)
        self.class_names = None  

        n = len(self.data)
        self.rgb_variances = []
        self.depth_variances = []

        if self.add_noise:
            if self.mode == 'train':
                self.rgb_variances = [float(np.random.randint(1, 12)) for _ in range(n)]
                self.depth_variances = [float(np.random.randint(1, 12)) for _ in range(n)]
                indices = random.sample(range(n), n // 2)
                for idx in indices:
                    self.rgb_variances[idx] = 1.0
                    self.depth_variances[idx] = 1.0

            elif self.mode == 'test':
                if self.val_half:
                    self.rgb_variances = [float(np.random.randint(1, 12)) for _ in range(n)]
                    self.depth_variances = [float(np.random.randint(1, 12)) for _ in range(n)]
                    indices = random.sample(range(n), n // 2)
                    for idx in indices:
                        self.rgb_variances[idx] = 1.0
                        self.depth_variances[idx] = 1.0
                else:
                    # 全部固定强度5
                    self.rgb_variances = [5.0] * n
                    self.depth_variances = [5.0] * n

            elif self.mode == 'testt':
                if self.val_modality == 'rgb':
                    rgb_rate = args.rgb_missing_rate
                    self.rgb_variances = [rgb_rate * 10 + 1] * n
                    self.depth_variances = [11.0] * n
                elif self.val_modality == 'depth':
                    depth_rate = args.depth_missing_rate
                    self.rgb_variances = [11.0] * n
                    self.depth_variances = [depth_rate * 10 + 1] * n
                elif self.val_modality == 'both':
                    rgb_rate = args.rgb_missing_rate
                    depth_rate = args.depth_missing_rate
                    self.rgb_variances = [rgb_rate * 10 + 1] * n
                    self.depth_variances = [depth_rate * 10 + 1] * n
                else:
                    raise ValueError(f"Unsupported val_modality: {val_modality}")
        else:
            self.rgb_variances = [1.0] * n
            self.depth_variances = [1.0] * n


    def __len__(self):
        return len(self.data)
    

    def _filter_frame_indices(self):
        """
        根据实际存在的图像文件，过滤 self.data 中每个样本的 frame_indices，
        仅保留 RGB 和 Depth 两个模态都存在的帧索引。
        """
        import glob
        filtered_data = []
        for sample in self.data:
            video_path = sample['video']
            rgb_dir = video_path
            depth_dir = video_path.replace('color', 'depth')

            # 获取 RGB 文件夹中所有 jpg 的数字索引
            rgb_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
            rgb_existing = {int(os.path.splitext(os.path.basename(f))[0]) for f in rgb_files}

            # 获取 Depth 文件夹中所有 jpg 的数字索引
            depth_files = glob.glob(os.path.join(depth_dir, "*.jpg"))
            depth_existing = {int(os.path.splitext(os.path.basename(f))[0]) for f in depth_files}

            # 原始帧索引集合
            original_indices = set(sample['frame_indices'])

            # 取三个集合的交集：标注范围内的、RGB存在的、Depth存在的
            valid_indices = sorted(original_indices & rgb_existing & depth_existing)
            # if(len(rgb_existing) != len(depth_existing)):
            #     # import ipdb; ipdb.set_trace();
            #     print(len(valid_indices))

            # print(len(valid_indices))

            if len(valid_indices) == 0:
                # 如果没有有效帧，打印警告并跳过该样本（或保留原索引但会在加载时报错）
                print(f"Warning: No valid frames for sample {sample['video_id']}, video path: {video_path}")
                continue  # 跳过该样本

            # 更新 frame_indices
            # if(valid_indices == sample['frame_indices']):
                # print(1111111111111)
            sample['frame_indices'] = valid_indices
            filtered_data.append(sample)

        self.data = filtered_data
        print(f"After filtering, dataset size: {len(self.data)}")


    def __getitem__(self, idx):
        sample_info = self.data[idx]
        video_path = sample_info['video']
        depth_path = video_path.replace('color', 'depth')
        # import ipdb; ipdb.set_trace();
        frame_indices = sample_info['frame_indices'] 
        label = sample_info['label']  # 从0开始才行，计算交叉熵的时候从0开始
        # print(label)

        rgb_var = self.rgb_variances[idx]
        depth_var = self.depth_variances[idx]

        # num_frame = self.args.num_frame
        # total_frames = len(frame_indices)
        # replace = total_frames < num_frame
        # selected_indices = np.random.choice(frame_indices, size=num_frame, replace=replace)
        # selected_indices.sort()

        num_frame = self.args.num_frame
        total_frames = len(frame_indices)

        if total_frames <= num_frame:
            # 不足 num_frame，直接补全
            replace = True
            selected_indices = np.random.choice(frame_indices, size=num_frame, replace=replace)
        else:
            # 分段随机采样
            segments = np.linspace(0, total_frames, num_frame + 1, dtype=int)
            selected_indices = []
            for i in range(num_frame):
                start = segments[i]
                end = segments[i + 1]
                if start == end:
                    idx = start
                else:
                    idx = np.random.randint(start, end)
                selected_indices.append(frame_indices[idx])
                
        selected_indices.sort()  # 保证升序

        
        # RGB 
        rgb_frames = []
        for i in selected_indices:
            img_path = os.path.join(video_path, f"{i:05d}.jpg")
            rgb_frames.append(pil_loader(img_path, 'RGB'))

        # Depth 
        depth_frames = []
        # depth_path = video_path.replace('color', 'depth')
        for i in selected_indices:
            img_path = os.path.join(depth_path, f"{i:05d}.jpg")
            depth_frames.append(pil_loader(img_path, 'Depth'))

        if self.mode == 'train':
            transform_rgb = transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(size=(224, 224)),
                AddMaskNoise(variance=rgb_var),      
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform_rgb = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                AddMaskNoise(variance=rgb_var),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        transform_depth = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            AddMaskNoise_spec(variance=depth_var),       
            transforms.ToTensor(),
        ])

        rgb_frames = [transform_rgb(img) for img in rgb_frames]
        depth_frames = [transform_depth(img) for img in depth_frames]

        rgb_tensor = torch.stack(rgb_frames, dim=0).permute(1, 0, 2, 3)
        depth_tensor = torch.stack(depth_frames, dim=0).permute(1, 0, 2, 3)

        return rgb_tensor, depth_tensor, label, rgb_var, depth_var








from torch.utils.data import DataLoader


def print_label_distribution(dataset, split_name):
    """统计并打印数据集中每个标签的样本数"""
    from collections import Counter
    labels = [sample['label'] for sample in dataset.data]
    counter = Counter(labels)
    print(f"\n{split_name} 标签分布 (共 {len(labels)} 个样本):")
    for label in sorted(counter.keys()):
        print(f"  类别 {label}: {counter[label]} 个样本")







class Args:
    """模拟 args 对象，包含数据集所需的参数"""
    random_seed = 42
    root_path = "/root/autodl-tmp/data/NVGesture/data"         
    rgb_missing_rate = 0.2
    depth_missing_rate = 0.3
    num_frame = 3

def test_dataset():
    args = Args()
    
    test_configs = [
        ('train', True),          # 训练模式，RGB模态，添加噪声
        ('test', True),            # 测试模式，RGB模态，添加噪声（固定强度5）
        # ('testt', 'Depth', True),         # testt模式，Depth模态，根据 val_modality 设置缺失
        # ('train',  False),          # 训练模式，不加噪声
    ]
    
    for mode, add_noise in test_configs:

        # 创建数据集实例
        dataset = NV_Noise_Mask(
            args=args,
            add_noise=add_noise,
            mode=mode,
            val_half=False,          
        )
        # dataset._fix_missing_frames(dry_run=False)
        # dataset._fix_missing_frames(dry_run=True)
        # print_label_distribution(dataset, mode)
        
        print(f"  数据集大小: {len(dataset)}")
        
        # # 获取第一个样本
        # rgb_tensor, depth_tensor, label, rgb_var, depth_var = dataset[0]
        # print(f"  张量形状: {rgb_tensor.shape, depth_tensor.shape}")   # 预期 (C, T, H, W)
        # print(f"  标签: {label}")
        # print(f"  RGB噪声强度: {rgb_var}, Depth噪声强度: {depth_var}")
        
        # # 测试 DataLoader（仅对训练模式做批量测试）
        # if mode == 'train':
        #     loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        #     batch = next(iter(loader))
        #     # batch 是一个元组 (video_tensor, label, rgb_var, depth_var)
        #     print(f"  Batch 视频形状: {batch[0].shape}")   # (B, C, T, H, W)
        #     print(f"  Batch 视频形状: {batch[1].shape}")   # (B, C, T, H, W)
        #     print(f"  Batch 标签形状: {batch[2].shape}")
        #     print(f"  Batch RGB方差: {batch[3]}")
        #     print(f"  Batch Depth方差: {batch[4]}")

if __name__ == "__main__":
    test_dataset()





