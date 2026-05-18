import streamlit as st
import os
import sys
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# 1. 确保能引用到模型代码
# 假设您的项目代码在 Modular_Latent_Space 文件夹下
PROJECT_ROOT = os.path.abspath("Modular_Latent_Space")
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "MPNN"))

# 导入您的模型类 (根据您的重现代码调整)
import sys
import os

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加 MPNN 模块所在路径到 Python 搜索路径
# 根据你的实际路径：Modular_Latent_Space-master/Modular_Latent_Space-master/MPNN
mpnn_parent_path = os.path.join(current_dir, 'Modular_Latent_Space-master', 'Modular_Latent_Space-master')
if mpnn_parent_path not in sys.path:
    sys.path.insert(0, mpnn_parent_path)

# 现在可以正常导入
from MPNN.mpnn import Toxicity_MPNN

# --- 缓存模型加载 ---
@st.cache_resource
def load_model():
    model = Toxicity_MPNN()
    # 路径指向您上传到 GitHub 的权重文件
    state_dict = torch.load("model_weights.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- 页面设计 ---
st.set_page_config(page_title="Crystal-Tox 预测", page_icon="💊")
st.title("🔬 Foundational Chemistry Model 线上预测")

st.markdown("""
本工具基于 **Emma King-Smith (2024)** 的论文重现，利用基础模型的潜空间进行毒性预测。
""")

smiles = st.text_input("输入分子 SMILES:", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C") # 默认咖啡因

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
                    # 注意：这里需要调用您在 predict_toxicity.py 中的预处理逻辑
                    # 比如将 SMILES 转换为 Graph 数据对象
                    # 示例伪代码：
                    # processed_data = my_preprocess_func(smiles)
                    # output = model(processed_data)
                    
                    # 占位符预测值
                    prediction = 4.25 
                    
                    st.metric(label="预测 log LD50 (mg/kg)", value=f"{prediction:.3f}")
                    
                    if prediction < 3.0:
                        st.error("结论：预测为高毒性")
                    else:
                        st.success("结论：预测为低毒性/安全")
                        
                except Exception as e:
                    st.error(f"模型推理失败: {e}")
        else:
            st.error("无效的 SMILES 字符串")
