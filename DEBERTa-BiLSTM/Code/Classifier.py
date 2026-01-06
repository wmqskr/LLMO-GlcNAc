import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import DebertaV2ForSequenceClassification
import torch.nn.functional as F

class DeBERTaBiLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.1):
        super(DeBERTaBiLSTMClassifier, self).__init__()
        
        # 加载deberta模型
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained("../deberta-v3-base")
        
        # 双向lstm
        self.lstm = nn.LSTM(input_size=768,  # 输入维度：deberta的隐藏层维度
                          hidden_size=hidden_size,  # lstm隐藏层维度
                          num_layers=3,  # lstm层数
                          bidirectional=True,  # 双向lstm
                          batch_first=True)  # 输入张量格式为(batch_size, seq_length, feature_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层（输入维度是lstm的输出维度，2倍hidden_size因为是双向的）
        self.fc = nn.Linear(hidden_size*2, 1)
        
        # Sigmoid激活函数（用于二分类）
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # 获取deberta的输出
        outputs= self.deberta(input_ids=input_ids, attention_mask=attention_mask, 
                           output_hidden_states=True, 
                           output_attentions=True)
        
        # deberta的last_hidden_state，形状为(batch_size, seq_len, hidden_size)
        outputs = outputs.hidden_states[12]
        # print(outputs.shape)
        
        # 将deberta的输出传入LSTM层
        lstm_output, _ = self.lstm(outputs)  # lstm_output形状(batch_size, seq_len, hidden_size * 2)
        
        # 对lstm的输出进行池化（这里使用均值池化）
        # pooled_output = lstm_output.mean(dim=1)  # 对seq_len维度取均值，输出形状(batch_size, hidden_size * 2)
        lstm_output = lstm_output.permute(0, 2, 1)  # 转置，输出形状(batch_size, hidden_size * 2, seq_len)
        k_size = lstm_output.shape[2]
        lstm_output = F.max_pool1d(lstm_output, kernel_size=k_size)
        pooled_output = lstm_output.squeeze(2)  # 去掉seq_len维度，输出形状(batch_size, hidden_size * 2)

        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 通过全连接层和sigmoid输出
        output = self.fc(pooled_output)
        output = self.sigmoid(output)
        
        return output
