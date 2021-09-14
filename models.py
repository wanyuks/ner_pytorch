# coding: utf-8

import torch
from torch import nn
from torchcrf import CRF


# 定义使用字向量+词向量的lstm模型
class NER(nn.Module):
    def __init__(self, hidden_size, char_emb, word_emb, tags_num, dropout_rate):
        super(NER, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.tag_num = tags_num
        self.char_emd = nn.Embedding.from_pretrained(char_emb, freeze=False, padding_idx=0)
        self.word_emd = nn.Embedding.from_pretrained(word_emb, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM((self.char_emd.embedding_dim + self.word_emd.embedding_dim), self.hidden_size,
                            batch_first=True, bidirectional=True)
        self.dp = nn.Dropout(self.dropout_rate)
        self.hidden2tag = nn.Linear(self.hidden_size * 2, self.tag_num)
        self.crf = CRF(self.tag_num, batch_first=True)

    def forward(self, char, word, mask=None):
        # char = inputs[0]
        # word = inputs[1]
        if not mask:
            mask = torch.ne(char, 0)
        embedding = torch.cat((self.char_emd(char), self.word_emd(word)), dim=-1)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dp(outputs)
        # 得到发射矩阵
        outputs = self.hidden2tag(outputs)
        return torch.LongTensor(self.crf.decode(outputs, mask))

    def log_likelihood(self, char, word, tags, mask=None):
        # char = inputs[0]
        # word = inputs[1]
        if not mask:
            mask = torch.ne(char, 0)
        embedding = torch.cat((self.char_emd(char), self.word_emd(word)), dim=-1)
        outputs, hidden = self.lstm(embedding)
        outputs = self.hidden2tag(outputs)
        outputs = self.dp(outputs)
        return -self.crf(outputs, tags, mask)
