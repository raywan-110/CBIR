# 基于局部敏感哈希与深度模型的CBIR系统
## contributor: **Borui Wan Yu Cai**
## 概述
> - **特征提取**：采用imagenet上预训练好的vggnet进行特征提取
> - **索引构建与优化**: 采用vggnet average features构建词袋
> - 检索算法：1. 线性查找 2.**带随机投影矩阵(特征降维，优化索引)的局部敏感哈希**(LSH)
## 数据库
> 采用9144张图片, 101类别的小型图像数据库进行初步测试.
## 结果
| 模型 | 描述子  | 检索算法  | 距离函数  | MMAP  |  average time  |
| ---- | ---- | ---- | ---- | ---- | ---- |
| vggnet | average features | Linear search | cosine |  |  |
| vggnet | average features | LSH(with random projection) | cosine | **98.6919%** | **0.0165s** |