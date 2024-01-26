# MyFramework

以 [EAkit](https://github.com/THU-KEG/EAkit) 为基础，研究基于图神经网络的知识图谱对齐，需要从 [HGCN](https://github.com/StephanieWyt/HGCN-JE-JR/) 获取 [数据集](https://drive.google.com/drive/folders/1mfaeLXdqFnOHLYBXiTHWI7MLwtfTgPYQ) 放至 ./data/ 下

## 环境

- python                         3.9.12
- torch                           1.12.0
- torch-cluster             1.6.0+pt112cpu
- torch-geometric           2.4.0
- torch-scatter             2.1.0+pt112cpu
- torch-sparse              0.6.16+pt112cpu          pypi_0    pypi
- torch-spline-conv         1.2.1+pt112cpu

## 主要模型

dataset:

- DBP15K

encoder:

- GCN-Align from EAkit
- KECG from https://github.com/THU-KEG/KECG
- CompGCN from http://github.com/malllabiisc/CompGCN
- R-GCN from https://github.com/JinheonBaek/RGCN
- mlp with nn.Linear()

decoder:

- Align from EAkit

## 运行命令

主要参数：

```
python run.py

--encoder default="GCN-Align", choices=['GCN-Align', 'RGCN', 'CompGCN', 'KECG', 'MLP']

--hiddens default="300,300,300" (including in_dim and out_dim)

--ent_init default='random', choices=['random', 'name']

-skip_conn default='none', choices=['none', 'highway', 'concatall', 'concat0andl', 'residual']

--activation default='F.elu', choices=['none', 'F.elu'])

--epoch default=10
```

example:

```
python run.py --encoder="GCN-Align" --hiddens="300,300,300" --ent_init="random" --skip_conn="none" --activation="none"
```
