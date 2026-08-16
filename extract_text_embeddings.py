import json
import torch
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 配置区域 =================
JSON_PATH = "morphology_descriptions.json"
OUTPUT_PT_PATH = "text_embeddings_raw.pt"
# 默认使用支持多语言的轻量级 MiniLM 句子编码模型
MODEL_NAME = r"D:/MyDesktop/论文/MyProject/models/MiniLM"
# ============================================

def main():
    print(f"正在加载轻量级句子编码器: {MODEL_NAME} ...")
    # 模型会自动下载到本地缓存
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"正在读取 {JSON_PATH} ...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    embeddings_dict = {}
    
    print("开始提取文本特征...")
    for sample_id_str, json_val_str in tqdm(data.items()):
        try:
            # 你生成的 JSON value 实际上是一个包含了换行符的 JSON 字符串，需要再次解析
            desc_dict = json.loads(json_val_str)
            
            # 将 JSON 的键值对拼接成一段连贯的自然语言，更有利于句子编码器理解上下文
            text_input = (
                f"纤维取向呈现{desc_dict.get('Orientation', '未知')}。 "
                f"纤维形貌及均匀度表现为{desc_dict.get('Fiber morphology/uniformity', '未知')}。 "
                f"缺陷方面，{desc_dict.get('Defects', '未知')}。 "
                f"网络结构方面，{desc_dict.get('Network/overlap/fusion', '未知')}。"
            )
            
        except json.JSONDecodeError:
            # 如果有个别解析失败，直接退化为使用原文本
            print(f"警告：样本 {sample_id_str} 内部 JSON 解析失败，将使用原始字符串。")
            text_input = json_val_str
            
        # 生成 384 维的特征向量
        with torch.no_grad():
            emb = model.encode(text_input, convert_to_tensor=True, show_progress_bar=False)
            
        # 存入字典，key 转为整数以便后续与 Dataset 匹配，value 放到 CPU 上节省显存
        embeddings_dict[int(sample_id_str)] = emb.cpu()
        
    # 将包含 235 个 384 维张量的字典持久化保存
    torch.save(embeddings_dict, OUTPUT_PT_PATH)
    print(f"提取完成！原始特征已保存至 {OUTPUT_PT_PATH}")
    
if __name__ == "__main__":
    main()