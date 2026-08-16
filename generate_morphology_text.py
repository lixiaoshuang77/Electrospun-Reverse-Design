import os
import glob
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn_extra.cluster import KMedoids
import pandas as pd
from tqdm import tqdm
import gc
# 引入 Qwen-VL 相关库
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置区域 =================
IMG_ROOT_DIR = r"D:\MyDesktop\论文\code\MatMCL\datasets\images\preprocessed"
CSV_PATH = r"D:\MyDesktop\论文\code\MatMCL\scripts\merged_sorted_data.csv"
QWEN_MODEL_PATH = r"D:\AI_Practice\model\Qwen\Qwen2-VL-2B-Instruct"
OUTPUT_JSON_PATH = "morphology_descriptions.json"
NUM_REPRESENTATIVE_PATCHES = 4  # 每个样本挑选 8 张代表性 Patch
# ============================================

class PatchSelector:
    """使用预训练 ResNet-18 和 K-Medoids 提取代表性 Patch"""
    def __init__(self, device='cuda'):
        self.device = device
        # 1. 加载预训练 ResNet-18，并去掉最后一层分类头，保留 512 维特征输出
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)#调用了 PyTorch 官方视觉库中的 ResNet-18 神经网络模型以及训练好的权重。
        self.encoder.fc = nn.Identity() #把 ResNet-18 的最后一层（全连接层fc）替换成了 nn.Identity()。PyTorch 里的一个“占位符”层，意思就是“什么都不做，直接把输入原封不动地输出”。
        self.encoder.to(self.device)#将神经网络模型的所有权重参数，搬运到指定的计算设备上。在 PyTorch 中，数据和模型必须在同一个硬件设备上才能进行矩阵乘法运算。
        self.encoder.eval()#将模型设置为“评估模式”，而非“训练模式”。神经网络中有些特殊的层（特别是 BatchNorm 批归一化层 和 Dropout 随机失活层），它们在“训练”和“测试”时的行为是完全不同的。让模型不要更新，不要改变行为
        
        # 2. 图像标准化（ImageNet 标准）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def select_top_k(self, img_paths, k=8):
        if len(img_paths) <= k:
            return img_paths  # 如果总数都不够 k 张，直接全要
            
        features = []
        valid_paths = []
        for p in img_paths:
            try:
                # 必须转 RGB，防二值化单通道报错
                img = Image.open(p).convert('RGB')
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                feat = self.encoder(tensor) # [1, 512]
                features.append(feat.cpu().numpy().flatten())
                valid_paths.append(p)
            except Exception as e:
                print(f"读取图片失败 {p}: {e}")
                
        if len(valid_paths) <= k:
            return valid_paths
            
        # 3. 使用 K-Medoids 聚类（K-Medoids 选出的中心点是真实的样本，而 K-Means 是虚拟坐标）
        # metric='cosine' 对高维视觉特征聚类效果更好
        kmedoids = KMedoids(n_clusters=k, metric='cosine', random_state=42)
        kmedoids.fit(features)
        
        # 获取聚类中心的索引
        center_indices = kmedoids.medoid_indices_
        selected_paths = [valid_paths[i] for i in center_indices]
        return selected_paths


from transformers import BitsAndBytesConfig

def load_qwen_model(model_path):
    """加载本地 Qwen2-VL 模型（8GB显存优化版）"""
    print("正在以 8-bit 量化模式加载 Qwen2-VL 模型以节省显存...")
    
    # 【修改点2】：配置 8-bit 量化，大幅度降低模型自身占用的显存
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto",
        quantization_config=quantization_config,
        # 【修改点3】：启用 PyTorch 2.0+ 原生的显存优化注意力机制
        attn_implementation="sdpa" 
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print("模型加载完成！")
    return model, processor

def generate_description(model, processor, img_paths):
    """构建 Prompt 并调用 Qwen2-VL 生成文本"""
    # 构建严格的系统提示词
    prompt_text = (
        "你是一个严谨的材料科学显微图像分析器。请观察提供的多张静电纺丝 SEM 图像贴片（这些图像来自同一个样本）。\n"
        "【绝对规则】\n"
        "1. 只描述肉眼可见的物理形貌，禁止推断工艺参数、材料成分或力学性能。\n"
        "2. 禁止估算绝对数值（如'直径300nm'），必须使用定性词汇（如粗、细、均匀、密集、稀疏）。\n"
        "3. 只输出 JSON 格式，不要包含任何额外的解释或 Markdown 代码块标记（如 ```json）。\n\n"
        "请严格输出以下包含 4 个字段的 JSON：\n"
        "{\n"
        "  \"Orientation\": \"描述纤维是否有明显方向性，还是随机分布\",\n"
        "  \"Fiber morphology/uniformity\": \"描述纤维表面的平整度及粗细是否均匀\",\n"
        "  \"Defects\": \"描述是否有明显的串珠现象(beading)、断裂或其他瑕疵\",\n"
        "  \"Network/overlap/fusion\": \"描述纤维之间的交叠密集程度，以及交叉点是否有融合现象\"\n"
        "}"
    )

    # Qwen2-VL 的多图对话结构构建
    content_list = []
    # 动态插入多张图片
    for path in img_paths:
        content_list.append({"type": "image", "image": path})
    # 插入文本 Prompt
    content_list.append({"type": "text", "text": prompt_text})

    messages = [
        {"role": "user", "content": content_list}
    ]

    # 准备输入给模型
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # 推理生成 (温度设为 0，确保输出稳定性和一致性)
    # 【修改点4】：清理 generation 参数，消除 Warning
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=256, 
            do_sample=False,  # 强制贪婪解码
            temperature=None, # 取消 temperature
            top_p=None,       # 取消 top_p
            top_k=None        # 取消 top_k
        )
        
    # 裁剪掉 prompt 部分，只保留生成的回复
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return output_text.strip()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. 获取所有有效的样本 ID (从你的 CSV 里面读)
    df = pd.read_csv(CSV_PATH)
    valid_ids = df['ID'].unique()
    print(f"总计需要处理 {len(valid_ids)} 个样本。")

    # 2. 初始化组件
    selector = PatchSelector(device=device)
    model, processor = load_qwen_model(QWEN_MODEL_PATH)

    results = {}
    
    # 支持断点续传（如果中间断了，再跑不会从头开始）
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
    # 3. 开始遍历生成
    for sample_id in tqdm(valid_ids, desc="生成形貌文字"):
        str_id = str(sample_id)
        if str_id in results:
            continue  # 已经处理过的跳过
            
        sample_folder = os.path.join(IMG_ROOT_DIR, str_id)
        if not os.path.exists(sample_folder):
            continue
            
        # 获取所有 patch 路径
        all_patches = glob.glob(os.path.join(sample_folder, '*.jpg'))
        
        # 步骤 1: 降采样，挑出 8 张最具代表性的
        selected_patches = selector.select_top_k(all_patches, k=NUM_REPRESENTATIVE_PATCHES)
        
        # 步骤 2: 喂给 Qwen 提取文本
        try:
            description = generate_description(model, processor, selected_patches)
            # 简单清理可能包含的 markdown json 标记
            description = description.replace("```json", "").replace("```", "").strip()
            
            results[str_id] = description
            
            # 每处理完一个，就保存一次
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            # === 强力显存回收机制（新增） ===
            import gc
            gc.collect()               # 强制 Python 回收内存
            torch.cuda.empty_cache()   # 清空 PyTorch 的显存缓存
            gc.collect()
            
        except Exception as e:
            print(f"样本 {sample_id} 处理出错: {e}")

    print(f"全部完成！结果已保存至 {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()