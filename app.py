import streamlit as st
import os
import sys
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import importlib.util
from collections import OrderedDict

# ========== 动态加载模块（绕过嵌套路径和连字符限制）==========
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 加载 Big_MPNN（底层模型）
mpnn_file = os.path.join(current_dir, 'Modular_Latent_Space-master', 'Modular_Latent_Space-master', 'MPNN', 'mpnn.py')
spec = importlib.util.spec_from_file_location("mpnn", mpnn_file)
mpnn_module = importlib.util.module_from_spec(spec)
sys.modules["mpnn"] = mpnn_module
spec.loader.exec_module(mpnn_module)

# 2. 加载 Toxicity_MPNN（毒性预测模型）
toxicity_file = os.path.join(current_dir, 'Modular_Latent_Space-master', 'Modular_Latent_Space-master', 'toxicity', 'mpnn_toxicity.py')
spec2 = importlib.util.spec_from_file_location("mpnn_toxicity", toxicity_file)
toxicity_module = importlib.util.module_from_spec(spec2)
sys.modules["mpnn_toxicity"] = toxicity_module
spec2.loader.exec_module(toxicity_module)

# 3. 加载 pytorch_toxicity（预处理函数）
pytorch_file = os.path.join(current_dir, 'Modular_Latent_Space-master', 'Modular_Latent_Space-master', 'toxicity', 'pytorch_toxicity.py')
spec3 = importlib.util.spec_from_file_location("pytorch_toxicity", pytorch_file)
pytorch_module = importlib.util.module_from_spec(spec3)
sys.modules["pytorch_toxicity"] = pytorch_module
spec3.loader.exec_module(pytorch_module)

Toxicity_MPNN = toxicity_module.Toxicity_MPNN
Toxic_NonToxic_Molecules = pytorch_module.Toxic_NonToxic_Molecules
canonicalize_smiles = pytorch_module.canonicalize_smiles
# ==============================================================

# 原子列表（来自 predict_toxicity.py）
ATOM_LIST = [6, 8, 7, 9, 17, 16, 15, 5, 29, 35, 14, 53, 30, 26, 27, 28, 48, 44, 42, 25, 47, 34, 46, 78, 50, 74,
             11, 3, 19, 23, 13, 79, 45, 82, 75, 77, 76, 51, 24, 33, 22, 32, 80, 92, 31, 63, 52, 12, 40, 65, 83,
             49, 64, 20, 66, 57, 60, 62, 39, 56, 68, 59, 58, 70, 55, 38, 41, 37, 73, 67, 21, 81, 71, 72, 90, 69,
             43, 4, 93, 94, 54, 95, 2, 10, 18, 98, 36, 96, 97, 91]

# 预训练模型路径
PRETRAINED_PATH = os.path.join(current_dir, 'Modular_Latent_Space-master', 'Modular_Latent_Space-master', 'MPNN', 'big_mpnn_no_delocalised_no_unknown_model')

# 最大分子大小（需要根据实际数据调整，先用一个合理值）
LONGEST_MOLECULE = 91

# --- 缓存模型加载 ---
@st.cache_resource
def load_model():
    model = Toxicity_MPNN(
        message_size=128,
        message_passes=3,
        atom_list=ATOM_LIST,
        pretrained_mpnn_path=PRETRAINED_PATH,
        longest_molecule=LONGEST_MOLECULE
    )
    
    # 加载微调后的权重
    finetuned_path = os.path.join(current_dir, 'finetuned_toxicity.pt')
    if os.path.exists(finetuned_path):
        state_dict = torch.load(finetuned_path, map_location=torch.device('cpu'))
        
        # 调试：显示权重键名（首次运行查看）
        # st.write("权重文件中的前10个键名:", list(state_dict.keys())[:10])
        # st.write("模型中的前10个键名:", list(model.state_dict().keys())[:10])
        
        # 检查是否需要转换键名（适配 gen_states 提取的格式）
        # 如果权重键名包含 'mpnn.' 前缀，需要去掉
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('mpnn.'):
                new_key = key[5:]  # 去掉 'mpnn.' 前缀
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 尝试加载转换后的权重
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            # 如果还是失败，尝试只加载 mpnn 部分的权重
            mpnn_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if 'mpnn.' in key:
                    new_key = key.split('mpnn.')[1]
                    mpnn_state_dict[new_key] = value
            
            model.mpnn.load_state_dict(mpnn_state_dict, strict=False)
            # toxicity_predictor 和 compress_mol 层保持随机初始化
    
    model.eval()
    return model

# --- SMILES 预处理函数 ---
def preprocess_smiles(smiles, longest_molecule=LONGEST_MOLECULE):
    """将 SMILES 字符串转换为模型输入格式"""
    # 创建临时 DataFrame
    df = pd.DataFrame({
        'canonical_smiles': [canonicalize_smiles(smiles)],
        'Y': [0.0]  # 占位标签
    })
    
    # 创建数据集实例
    dataset = Toxic_NonToxic_Molecules(df, longest_molecule)
    
    # 获取第一个样本
    (mol_features, mol_matrices), _ = dataset[0]
    
    # 添加 batch 维度
    mol_features = mol_features.unsqueeze(0)
    mol_matrices = mol_matrices.unsqueeze(0)
    
    return mol_features, mol_matrices

# --- 页面设计 ---
st.set_page_config(page_title="Crystal-Tox 预测", page_icon="💊")
st.title("🔬 Foundational Chemistry Model 线上预测")

st.markdown("""
本工具基于 **Emma King-Smith (2024)** 的论文重现，利用基础模型的潜空间进行毒性预测。
""")

smiles = st.text_input("输入分子 SMILES:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  # 默认咖啡因

if st.button("开始预测"):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # 展示结构
            img = Draw.MolToImage(mol)
            st.image(img, caption="分子结构预览")
            
            # 推理逻辑
            model = load_model()
            with st.spinner('计算中...'):
                try:
                    # 预处理 SMILES
                    mol_features, mol_matrices = preprocess_smiles(smiles)
                    
                    # 模型推理
                    with torch.no_grad():
                        prediction = model(mol_matrices, mol_features)
                        prediction = prediction.item()
                    
                    st.metric(label="预测 log LD50 (mg/kg)", value=f"{prediction:.3f}")
                    
                    if prediction < 3.0:
                        st.error("结论：预测为高毒性")
                    else:
                        st.success("结论：预测为低毒性/安全")
                        
                except Exception as e:
                    st.error(f"模型推理失败: {e}")
                    st.error(f"错误详情: {str(e)}")
        else:
            st.error("无效的 SMILES 字符串")
