import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, DebertaV2ForSequenceClassification

class DeBERTaBiGRUClassifier(nn.Module):
    def __init__(self, deberta_path="../deberta-v3-base", hidden_size=256, dropout=0.1):
        super(DeBERTaBiGRUClassifier, self).__init__()
        
        # 加载 DeBERTa 预训练模型
        self.deberta = DebertaV2ForSequenceClassification.from_pretrained(deberta_path)
        
        # 初始化 BiGRU
        self.gru = nn.GRU(
            input_size=768,  # DeBERTa 的输出维度
            hidden_size=hidden_size,                    # GRU 的隐藏层维度
            num_layers=2,                               # 堆叠两层 GRU
            bidirectional=True,                         # 使用双向 GRU
            batch_first=True                            # 输入张量格式为 (batch_size, seq_len, embedding_dim)
        )
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 分类层（输入维度是双向GRU输出的2倍hidden_size，输出一个数用于二分类）
        self.fc = nn.Linear(hidden_size * 2, 1)
        
        # Sigmoid 激活函数将输出映射到 0-1 范围（用于二分类）
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        """
        input_ids:       Tensor of shape (batch_size, seq_len)，每个 token 的索引
        attention_mask:  Tensor of shape (batch_size, seq_len)，mask 指示哪些 token 是有效的（1）或 padding（0）
        """
        
        # Step 1: 获取 DeBERTa 输出
        deberta_output = self.deberta(input_ids=input_ids, attention_mask=attention_mask,
                                      output_hidden_states=True, 
                                      output_attentions=True)
        
        last_hidden = deberta_output.hidden_states[12]  # shape: (batch_size, seq_len, hidden_size)
        
        # Step 2: 将 DeBERTa 的输出送入 BiGRU
        gru_output, _ = self.gru(last_hidden)  # shape: (batch_size, seq_len, hidden_size * 2)
        
        # Step 3: 使用 1D 最大池化将每个样本的序列输出压缩为一个固定维度的向量
        # 转置为 (batch_size, hidden_size * 2, seq_len)，便于后续做 1D pooling
        gru_output = gru_output.permute(0, 2, 1)
        # 使用 max_pool1d 在序列维度做池化，得到 (batch_size, hidden_size * 2, 1)
        pooled_output = F.max_pool1d(gru_output, kernel_size=gru_output.shape[2])
        # 去掉多余的维度，变成 (batch_size, hidden_size * 2)
        pooled_output = pooled_output.squeeze(2)
        # Step 4: Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Step 5: 全连接层 + Sigmoid 输出概率
        logits = self.fc(pooled_output)         # (batch_size, 1)
        probs = self.sigmoid(logits)            # 输出值范围为 0~1，适用于二分类
        
        return probs  # shape: (batch_size, 1)
