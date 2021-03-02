# 基于内容的图像检索系统

## 概述

> - **特征提取**：采用Resnet101+GeM Pooling上进行特征提取(预训练的网络)
> - **索引构建与优化**: 采用矢量量化(PQ)+倒排索引(IVF)构建特征数据库，并进行检索。(利用Faiss库实现此功能)

**The Oxford Buildings Dataset测评结果**

|  mAP  | average time(TiTan Xp) |
| :---: | :--------------------: |
| 0.843 |         0.14s          |



## 运行环境

python 3.6

pytorch 1.3.1

faiss-cpu 1.6.5

在linux服务器上完成特征库提取、性能测评，在windows10上运行交互界面与功能展示。



## 数据集准备

### Oxford5k

下载[Oxford5k](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/)数据集及标注后，按照如下文件结构放置

```
oxbuild_images
|_ all_souls_000000.jpg
|_ ...
|_ worcester_000198.jpg

eval_oxford5k
|_ gt_files_170407
|  |_ all_souls_1_good.txt
|  |_ all_souls_1_junk.txt
|  |_ ...
```

### 50000图像数据库

我们将自行构建的图像数据库分类并按照如下文件结构放置

```
database
|_ 001.ak47
|  |_ <id-1>.jpg
|  |_ ...
|  |_ <id-n>.jpg
|_ 002.american-flag
|  |_ <id-1>.jpg
|  |_ ...
|  |_ <id-n>.jpg
|_ ...
```



## 生成特征库

在`model.py`中指定：

```python
isOxford=True  # 指定oxford5k数据集
# isOxford=False  # 指定database数据集
```

并在`DB.py`中对应指定：

```python
# oxford5k
DB_dir = '..\\oxbuild_images'  # 数据集所在文件夹
DB_csv = '..\\oxbuild_images.csv'  # 将所有图片的路径保存在.csv中

# # database
# DB_dir = '..\\database'
# DB_csv = '..\\database.csv'
```

要生成特征库和对应的特征索引，运行：

```
python model.py
```

最终生成的特征库和索引文件如下所示

```
cache
|_ res101_AP_GeM-oxf-dict
|_ res101_AP_GeM-oxf-indexIPQ
|_ res101_AP_GeM-oxf-vec
|_ res101_AP_GeM-database-dict
|_ res101_AP_GeM-database-indexIPQ
|_ res101_AP_GeM-database-vec
```



## 运行交互界面

1. 在windows 10 系统上运行`app.py`

2. 指定检索的数据集（`oxford5k`或`database`）

3. 将待查询的图片拖拽进指定区域内，或浏览上传
4. 点击**开始检索**按钮即可展示检索结果



## 性能测评

该项目在Oxford5k数据集上测评**mAP**与**每张图片的检索时间**。

### 生成ranked_list

进入`src/`目录，运行

```
python eval_oxford5k.py
```

将在`eval_oxford5k/ranked_list/`目录下生成55个query图片的ranked_list，如下所示：

```
eval_oxford5k
|_ ranked_list
|  |_ all_souls_1.txt
|  |_ ...
|  |_ radcliffe_camera_5.txt
```

同时打印出**检索总时间**以及**平均每张图片所需时间**。

### 计算AP与mAP

进入`eval_oxford5k/`目录，运行

```
python compute_mAP.py
```

将在该目录下生成`AP_list.txt`。其中记录了每个query的AP，最后一行是55个AP的平均值mAP，如下所示

```
all_souls_1	0.7677968608505781
all_souls_2	0.7413266519289257
all_souls_3	0.8199504211167984
all_souls_4	0.9264498388831742
all_souls_5	0.9061729807755552
ashmolean_1	0.9239986724039997
ashmolean_2	0.8786306277732049
ashmolean_3	0.850630505882592
ashmolean_4	0.9331057090345016
ashmolean_5	0.7795071401326593
...
mAP	0.8427707479798218
```

