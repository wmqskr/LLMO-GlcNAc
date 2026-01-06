import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from train import train_and_evaluate_model_with_cv



# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

  
#提取序列
def remove_name(data):
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new


#读取数据
with  open("../dataset/ESM2_NearMiss/S/train_101_S.fasta") as f:
    pos_neg_Data= f.readlines()
    pos_neg_Data = [s.strip() for s in pos_neg_Data]
print(len(pos_neg_Data))
print("数据读取完成")
print("———————————————————————————————————————————————————")

pos_neg_Data = remove_name(pos_neg_Data)

print(len(pos_neg_Data),len(pos_neg_Data[0]))
print("序列提取完成")
print("———————————————————————————————————————————————————")


#定义标签
pos_neg_label = np.concatenate([np.ones(1707), np.zeros(1707)], axis=0)  #竖向拼接
print(pos_neg_label.shape)
print("标签定义完成")
print("———————————————————————————————————————————————————")

# 10折交叉验证
train_and_evaluate_model_with_cv(pos_neg_Data, pos_neg_label)
print("模型训练完成")
print("S101")

