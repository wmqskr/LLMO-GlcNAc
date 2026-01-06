import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DebertaV2ForSequenceClassification, ModernBertForSequenceClassification
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, f1_score, precision_score
from torch.utils.data import TensorDataset
from DeBERTaBiGRUClassifier import *
from DeBERTaBiLSTMClassifier import *
import warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()



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
with  open("../dataset/ESM2_NearMiss/T/test_101_T.fasta") as f:
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
pos_neg_label = np.concatenate([np.ones(270), np.zeros(957)], axis=0)  #竖向拼接
print(pos_neg_label.shape)
print("标签定义完成")
print("———————————————————————————————————————————————————")


def test_model(test_data, test_label):
    # 设备设置
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)

    # 初始化 tokenizer 和模型
    modernberttokenizer = AutoTokenizer.from_pretrained("../modernbert")
    debertatokenizer = AutoTokenizer.from_pretrained("../deberta-v3-base")

    modernbert_model = ModernBertForSequenceClassification.from_pretrained("../modernbert", num_labels=1).to(device)
    modernbert_model.load_state_dict(torch.load('../model/11T101_modernbert_model.pth', map_location='cuda:2'))
    debertabilstm_model = DeBERTaBiLSTMClassifier().to(device)
    debertabilstm_model.load_state_dict(torch.load('../model/7T101_debertabilstm_model.pth', map_location='cuda:2'))
    debertabigru_model = DeBERTaBiGRUClassifier().to(device)
    debertabigru_model.load_state_dict(torch.load('../model/8T101_debertabigru_model.pth', map_location='cuda:2'))

    modernbert_model.eval()
    debertabilstm_model.eval()
    debertabigru_model.eval()

    # 数据预处理
    modernbertencoded_texts = modernberttokenizer(test_data, padding=True, truncation=True, return_tensors='pt')
    modernbertinput_ids = modernbertencoded_texts['input_ids'].to(device)
    modernbertattention_mask = modernbertencoded_texts['attention_mask'].to(device)
    labels = torch.tensor(test_label, dtype=torch.float32).unsqueeze(1).to(device)

    debertaencoded_texts = debertatokenizer(test_data, padding=True, truncation=True, return_tensors='pt')
    debertainput_ids = debertaencoded_texts['input_ids'].to(device)
    debertaattention_mask = debertaencoded_texts['attention_mask'].to(device)
    labels = torch.tensor(test_label, dtype=torch.float32).unsqueeze(1).to(device)

    # 创建 DataLoader
    modernberttest_dataset = TensorDataset(modernbertinput_ids, modernbertattention_mask, labels)
    modernberttest_loader = DataLoader(modernberttest_dataset, batch_size=33, shuffle=False)

    debertatest_dataset = TensorDataset(debertainput_ids, debertaattention_mask, labels)
    debertatest_loader = DataLoader(debertatest_dataset, batch_size=33, shuffle=False)

    # 在测试数据上进行推断
    modernbert_all_preds = []
    modernbert_all_probs = []
    modernbert_all_labels = []

    debertabilstm_all_preds = []
    debertabilstm_all_probs = []
    debertabilstm_all_labels = []

    debertabigru_all_preds = []
    debertabigru_all_probs = []
    debertabigru_all_labels = []

    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in modernberttest_loader:
            outputs = modernbert_model(batch_input_ids, batch_attention_mask, 
                            labels=batch_labels, return_dict=True)
            outputs = outputs.logits
            batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
            batch_probs = outputs.cpu().detach().numpy()
            modernbert_all_preds.extend(batch_preds.tolist())
            modernbert_all_probs.extend(batch_probs.tolist())
            modernbert_all_labels.extend(batch_labels.detach().cpu().numpy())
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in debertatest_loader:
            outputs = debertabilstm_model(batch_input_ids, batch_attention_mask)
            batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
            batch_probs = outputs.cpu().detach().numpy()
            debertabilstm_all_preds.extend(batch_preds.tolist())
            debertabilstm_all_probs.extend(batch_probs.tolist())
            debertabilstm_all_labels.extend(batch_labels.detach().cpu().numpy())
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask, batch_labels in debertatest_loader:
            outputs = debertabigru_model(batch_input_ids, batch_attention_mask)
            batch_preds = (outputs >= 0.5).squeeze().cpu().numpy()
            batch_probs = outputs.cpu().detach().numpy()
            debertabigru_all_preds.extend(batch_preds.tolist())
            debertabigru_all_probs.extend(batch_probs.tolist())
            debertabigru_all_labels.extend(batch_labels.detach().cpu().numpy())


    # === Step 1: 转为 numpy 数组 ===
    modernbert_probs = np.array(modernbert_all_probs)
    debertabilstm_probs = np.array(debertabilstm_all_probs)
    debertabigru_probs = np.array(debertabigru_all_probs)

    # Labels 只用一个（因为 test_loader 是一样的）
    all_labels = np.array(debertabilstm_all_labels)  # or any one of the label lists

    # === Step 2: 统一维度 ===
    # 如果是 (N, 1)，转为 (N,)
    if modernbert_probs.ndim > 1:
        modernbert_probs = modernbert_probs.squeeze()
    if debertabilstm_probs.ndim > 1:
        debertabilstm_probs = debertabilstm_probs.squeeze()
    if debertabigru_probs.ndim > 1:
        debertabigru_probs = debertabigru_probs.squeeze()
    if all_labels.ndim > 1:
        all_labels = all_labels.squeeze()

    # === Step 3: 集成（取平均）===
    ensemble_probs = modernbert_probs*0.5+ debertabilstm_probs*0.25 + debertabigru_probs*0.25
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    # 计算指标
    mcc = matthews_corrcoef(all_labels, ensemble_preds)
    auc = roc_auc_score(all_labels, ensemble_probs)
    f1 = f1_score(all_labels, ensemble_preds)
    precision = precision_score(all_labels, ensemble_preds)
    TP = TN = FP = FN = 0
    for i in range(len(all_labels)):
        if ensemble_preds[i] == 1 and all_labels[i] == 1:
            TP += 1
        elif ensemble_preds[i] == 0 and all_labels[i] == 0:
            TN += 1 
        elif ensemble_preds[i] == 1 and all_labels[i] == 0:
            FP += 1
        else:
            FN += 1
    acc = (TP + TN) / (TP + TN + FP + FN)
    sn = TP / (TP + FN)
    sp = TN / (TN + FP)
    print(f"Test Sn: {sn:.4f}, Sp: {sp:.4f}, MCC: {mcc:.4f}, Acc: {acc:.4f}, Precision: {precision:.4f}, f1: {f1:.4f}, AUC: {auc:.4f}")

test_model(pos_neg_Data, pos_neg_label)
