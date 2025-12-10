

## 📖 项目简介 (Project Overview)

本项目是机器学习课程的期末作业，旨在基于 [ReChorus](https://github.com/THUwangcy/ReChorus) 推荐系统框架，复现并分析 **XSimGCL** (Extremely Simple Graph Contrastive Learning) 算法。

我们选取了两个具有代表性的数据集（**MovieLens-1M** 和 **Grocery & Gourmet Food**），将 XSimGCL 与基线模型 **LightGCN** 和 **DirectAU** 进行了对比实验。实验重点验证了在数据稀疏场景下，基于噪声增强的对比学习方法相比于传统 GCN 方法的性能提升。

## 👥 小组成员 (Team Members)

* **小组编号:** 24
* **成员 1:** 刘庭志 (23330082)
* **成员 2:** 金子豪 (23330051)

## 🛠️ 环境依赖与部署 (Environment & Setup)

本项目基于 Python 和 PyTorch 开发，建议在 Anaconda 环境下运行。

### 1. 环境要求
* **Python** >= 3.7
* **PyTorch** >= 1.7.0 (建议使用 GPU 版本以加速训练)
* **CUDA** (如果使用 GPU)
* **ReChorus 依赖库:** numpy, pandas, scipy, sklearn

### 2. 部署步骤

```bash
# git clone [YOUR_REPO_LINK]
cd ReChorus

# 2. 创建并激活虚拟环境 (可选但推荐)
conda create -n rechorus python=3.8
conda activate rechorus

# 3. 安装依赖
# 如果已有 requirements.txt
pip install -r requirements.txt
# 或者手动安装核心库
pip install torch numpy pandas scipy scikit-learn

````


## 📂 数据集准备 (Data Preparation)

请确保原始数据文件已正确预处理并放置在 `data/` 目录下，目录结构应如下所示：

Plaintext

```
ReChorus/
├── data/
│   ├── MovieLens_1M/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── ...
│   └── Grocery_and_Gourmet_Food/  (注意文件夹名称匹配 dataset 参数)
│       ├── train.csv
│       ├── test.csv
│       └── ...
├── src/
│   ├── main.py
│   ├── models/
│   │   ├── general/
│   │   │   ├── LightGCN.py
│   │   │   ├── DirectAU.py
│   │   │   └── XSimGCL.py  <-- 我们的核心复现代码
│   └── ...
```

## 🚀 实验复现命令 

> ⚠️ Windows 用户重要提示：
> 
> 由于 PyTorch 在 Windows 系统下的多进程 DataLoader 存在兼容性问题，必须在命令中添加 --num_workers 0，否则程序会在预测阶段报错崩溃。Linux 用户可将其设置为 4 或 5 以提高速度。

### 1. Grocery & Gourmet Food (稀疏数据集)

该数据集极其稀疏，LightGCN 需要特定的超参数（降低学习率、增加正则化）才能收敛。

- **LightGCN (Baseline)**
    
    - _参数亮点:_ `lr=1e-4`, `l2=1e-4`
        
    
    Bash
    
    ```
    python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --emb_size 64 --lr 1e-4 --l2 1e-4 --random_seed 0 --gpu 0 --regenerate 1 --num_workers 0
    ```
    
- **DirectAU (Baseline)**
    
    - _参数亮点:_ `lr=1e-3`, `l2=1e-6`, `gamma=1`
        
    
    Bash
    
    ```
    python src/main.py --model_name DirectAU --dataset Grocery_and_Gourmet_Food --emb_size 64 --lr 1e-3 --l2 1e-6 --gamma 1 --random_seed 0 --gpu 0 --num_workers 0
    ```
    
- **XSimGCL (Ours)**
    
    - _参数亮点:_ `eps=0.1`, `tau=0.1`
        
    
    Bash
    
    ```
    python src/main.py --model_name XSimGCL --dataset Grocery_and_Gourmet_Food --emb_size 64 --lr 1e-3 --l2 1e-6 --eps 0.1 --tau 0.1 --random_seed 0 --gpu 0 --num_workers 0
    ```
    

### 2. MovieLens-1M (稠密数据集)

该数据集较稠密，使用标准参数即可。

- **LightGCN (Baseline)**
    
    Bash
    
    ```
    python src/main.py --model_name LightGCN --dataset MovieLens_1M --emb_size 64 --lr 1e-3 --l2 1e-6 --random_seed 0 --gpu 0 --num_workers 0
    ```
    
- **DirectAU (Baseline)**
    
    Bash
    
    ```
    python src/main.py --model_name DirectAU --dataset MovieLens_1M --emb_size 64 --lr 1e-3 --l2 1e-6 --gamma 1 --random_seed 0 --gpu 0 --num_workers 0
    ```
    
- **XSimGCL (Ours)**
    
    Bash
    
    ```
    python src/main.py --model_name XSimGCL --dataset MovieLens_1M --emb_size 64 --lr 1e-3 --l2 1e-6 --eps 0.1 --tau 0.1 --random_seed 0 --gpu 0 --num_workers 0
    ```
    

## 📊 实验结果摘要 (Results Summary)

基于上述命令运行得到的测试集 (Test Set) Top-20 性能指标：

|**Dataset**|**Model**|**Recall@20**|**NDCG@20**|**结论 (Conclusion)**|
|---|---|---|---|---|
|**MovieLens-1M**|LightGCN|0.8181|0.4094|稠密数据上表现强劲|
|(稠密)|DirectAU|0.6780|0.3587|表现不如 LightGCN|
||**XSimGCL**|**0.8189**|**0.4127**|**微弱提升 (+0.8%)**|
|**Grocery**|LightGCN|0.4185|0.1752|稀疏数据下效果一般|
|(稀疏)|DirectAU|0.5185|0.2842|优于 LightGCN|
||**XSimGCL**|**0.6293**|**0.3415**|**巨大提升 (+94.9%)**|

## 📄 参考文献 (References)

[1] Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2023). **XSimGCL: Towards extremely simple graph contrastive learning for recommendation.** _IEEE Transactions on Knowledge and Data Engineering_, 36(2), 913-926.

[2] Wang, C., Zhang, M., Ma, W., Liu, Y., & Ma, S. (2020). **ReChorus: A Comprehensive and Modular Recommendation Framework.** _Proceedings of the 43rd International ACM SIGIR Conference_.
