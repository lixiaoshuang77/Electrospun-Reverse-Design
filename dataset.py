import os
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ElectrospunDataset(Dataset):
    def __init__(self, csv_path, img_root_dir, phase='train', m_patches=16):
        """
        静电纺丝多模态数据集
        :param csv_path: 你的 merged_sorted_data.csv 路径
        :param img_root_dir: 存放图片文件夹的根目录，例如 'D:/MyDesktop/论文/code/MatMCL/datasets/images/preprocessed'
        :param phase: 'train' 或 'val' / 'test'。决定采样策略和数据增强。
        :param m_patches: 训练时每个样本抽取的 patch 数量，默认 16。
        """
        self.img_root_dir = img_root_dir
        self.phase = phase
        self.m_patches = m_patches
        
        # 1. 核心逻辑：读取 CSV 并将 2 行（横/纵）合并为 1 行（10维性能）
        self.samples = self._process_csv(csv_path)
        
        # 2. 定义严格受限的数据增强与预处理 (依据 4.4 节)
        # ImageNet 标准均值和方差
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # 严禁 RandomRotation 和 RandomHorizontalFlip！会破坏取向物理意义
                transforms.ColorJitter(brightness=0.1, contrast=0.1), # 仅允许轻微扰动
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def _process_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        processed_data = []
        
        # 按材料 ID 分组，每个 ID 应该有 dir=0 和 dir=1 两条记录
        grouped = df.groupby('ID')
        
        for sample_id, group in grouped:
            # 校验：确保该 ID 有完整的横向和纵向数据
            if len(group) != 2:
                print(f"警告: ID {sample_id} 的数据行数不为 2，已跳过。")
                continue
                
            # 分离横向(0)和纵向(1)
            row_dir0 = group[group['dir'] == 0].iloc[0]
            row_dir1 = group[group['dir'] == 1].iloc[0]
            
            # 工艺参数 X (6维)。同一材料两行工艺是一样的，取 row_dir0 即可
            X = row_dir0[['f', 'c', 'v', 'r', 't', 'w']].values.astype(np.float32)
            
            # 力学性能 P (10维 = 5维横向 + 5维纵向)
            props = ['fracture', 'elongation', 'elastic modulus', 'tangent modulus', 'yield']
            P_0 = row_dir0[props].values.astype(np.float32)
            P_1 = row_dir1[props].values.astype(np.float32)
            P = np.concatenate([P_0, P_1]) # 拼接成 10 维
            
            # 校验图片文件夹是否存在 (过滤掉 CSV 里有，但图片文件夹丢失的异常数据)
            sample_folder = os.path.join(self.img_root_dir, str(sample_id))
            if not os.path.exists(sample_folder):
                continue
                
            processed_data.append({
                'ID': sample_id,
                'X': X,
                'P': P,
                'img_folder': sample_folder
            })
            
        print(f"[{self.phase}] 成功加载并合并了 {len(processed_data)} 个有效材料样本。")
        return processed_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 提取数值标签
        X = torch.tensor(sample['X'], dtype=torch.float32)
        P = torch.tensor(sample['P'], dtype=torch.float32)
        
        # 2. 图像 Patch 采样
        patch_paths = glob.glob(os.path.join(sample['img_folder'], '*.jpg'))
        if len(patch_paths) == 0:
            raise RuntimeError(f"ID {sample['ID']} 的文件夹中没有找到 jpg 图片！")
            
        if self.phase == 'train':
            # 训练阶段：随机抽取 M=16 个（random.choices 默认有放回采样，满足不足 16 个的兜底逻辑）
            selected_paths = random.choices(patch_paths, k=self.m_patches)
        else:
            # 验证/测试阶段：使用全部 Patch
            selected_paths = patch_paths
            
        # 3. 读取并转换图像
        images = []
        for p in selected_paths:
            # 必须 convert('RGB')！因为 Canny 和 大津法 处理后的图可能是单通道灰度图
            img = Image.open(p).convert('RGB')
            img_tensor = self.transform(img)
            images.append(img_tensor)
            
        # 将 list 堆叠成 Tensor。
        # 训练时 shape: [16, 3, 224, 224]
        # 测试时 shape: [N_all, 3, 224, 224] (N_all 可能是 60)
        I_patches = torch.stack(images, dim=0)
        
        return {
            'ID': sample['ID'],
            'X': X,         # [6]
            'I': I_patches, # [M, 3, 224, 224]
            'P': P          # [10]
        }