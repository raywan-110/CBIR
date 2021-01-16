# 基于PQ+IVF与深度模型的CBIR系统
## contributor: **Borui Wan Yu Cai**
## 概述
> - **特征提取**：采用Resnet101+GeM Pooling上进行特征提取(预训练的网络)
> - **索引构建与优化**: 采用矢量量化(PQ)+倒排索引(IVF)构建特征数据库，并进行检索。(利用Faiss库实现此功能)
## 数据库
> 采用51805张图片, 361类别的小型图像数据库进行初步测试.
## Oxford Building测试结果
| 模型 | 池化方法  | 检索算法  | 距离函数  | MMAP  |  average time(TiTan Xp)  |
| ---- | ---- | ---- | ---- | ---- | ---- |
| resnet101 | GeM Pooling | PQ+IVF | L2 | 84.28% | 0.01s |
- - - 
(注：图像数据库与特征数据库需要自行构建)