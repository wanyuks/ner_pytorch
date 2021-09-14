# coding:utf-8
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vectors
import jieba_fast
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


# 构建词典和向量映射
def load_embedding(embedding_file):
    token2id = {}
    emb = []
    embedding = Vectors(embedding_file)

    token2id["<pad>"] = len(token2id)
    token2id["<unk>"] = len(token2id)
    emb.append(np.random.normal(loc=0., scale=0.05, size=embedding.vectors.shape[1]))
    emb.append(np.random.normal(loc=0., scale=0.05, size=embedding.vectors.shape[1]))
    for c in embedding.stoi.keys():
        if c != "<unk>":
            token2id[c] = len(token2id)
            emb.append(embedding[c].numpy())
    return token2id, torch.tensor(emb, dtype=torch.float32)


def load_data(file, char_token2id, word_token2id, load_test_data=False, label_json="label.json"):
    inputs_char = []
    inputs_word = []
    labels = []
    char_level = []
    word_level = []
    tmp_label = []
    if load_test_data:
        with open(label_json, "r", encoding="utf-8") as f:
            label2id = json.load(f)
    else:
        label2id = {"<pad>": 0}
    origin_text = ""
    with open(file, encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.rstrip().split()
            if line:
                char_level.append(char_token2id.get(line[0], 1))
                origin_text += line[0]
                if line[1] not in label2id:
                    label2id[line[1]] = len(label2id)
                tmp_label.append(label2id[line[1]])
            else:
                if char_level:
                    inputs_char.append(char_level)
                    labels.append(tmp_label)

                    # 对于word level，为了保证和char level的维度对其，将词*词的长度，例如，天气不错，分词结果为
                    # [天气， 不错]， 那么word level的输入为[天气， 天气， 不错， 不错]
                    words = jieba_fast.cut(origin_text)
                    for w in words:
                        word_level += [word_token2id.get(w, 1)] * len(w)
                    inputs_word.append(word_level)
                    assert len(char_level) == len(word_level)
                    char_level = []
                    tmp_label = []
                    word_level = []
                    origin_text = ""
    if not load_test_data:
        with open(label_json, "w", encoding="utf-8") as f:
            json.dump(label2id, f)
    return inputs_char, inputs_word, labels, label2id


class NERDataset(Dataset):
    def __init__(self, inputs_char, inputs_word, labels, maxlen=256, padding=True, pad_token=0, truncation=True):
        self.inputs_char = inputs_char
        self.inputs_word = inputs_word
        self.labels = labels
        assert len(inputs_char) == len(inputs_word) == len(labels)
        self.maxlen = maxlen
        self.padding = padding
        self.pad_token = pad_token
        self.truncation = truncation

    def __getitem__(self, index):
        chars = self.inputs_char[index]
        words = self.inputs_word[index]
        assert len(chars) == len(words)
        label = self.labels[index]
        if chars and words:
            if self.padding:
                if len(chars) < self.maxlen:
                    chars = np.append(chars, [self.pad_token] * (self.maxlen - len(chars)))
                    words = np.append(words, [self.pad_token] * (self.maxlen - len(words)))
                    label = np.append(label, [self.pad_token] * (self.maxlen - len(label)))
            if self.truncation:
                if len(chars) > self.maxlen:
                    chars = chars[:self.maxlen]
                    words = words[:self.maxlen]
                    label = label[:self.maxlen]
            # inputs = torch.LongTensor([chars, words])
            chars = torch.LongTensor(chars)
            words = torch.LongTensor(words)
            return (chars, words), torch.LongTensor(label)

    def __len__(self):
        return len(self.inputs_char)


if __name__ == '__main__':
    char_file = "/home/wangjian/workspace/ner/train/workspace/test_proj_aspect_sentiment/" \
                "embedding/test_proj_glove_char.txt"
    word_file = "/home/wangjian/workspace/ner/train/workspace/test_proj_aspect_sentiment/" \
                "embedding/test_proj_glove_word.txt"
    char_token2id, char_emb = load_embedding(char_file)
    word_token2id, word_emb = load_embedding(word_file)
    train_input_chars, train_inputs_words, train_labels, _ = load_data("../data/train/3squirrels/aspect.train",
                                                                    char_token2id, word_token2id)
    ner_dataset = NERDataset(train_input_chars, train_inputs_words, train_labels)
    dataloader = DataLoader(dataset=ner_dataset, batch_size=64, shuffle=True)
    for idx, ((char, word), label) in enumerate(dataloader):
        print(label)
        break
