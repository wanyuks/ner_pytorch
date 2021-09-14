from models import NER
from dataloader import *
import torch
import pandas as pd
from conlleval import evaluate, return_report
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda:0")
hidden_size = 256
dropout_rate = 0.2
epochs = 20
learning_rate = 2e-4


def train_step(model, epochs, train_dataloader, test_dataloader, optimizer,
               id2token, id2label, save_path):
    model.train()
    best_f1 = 0.
    for epoch in range(epochs):
        train_loss = 0.
        for idx, (inputs, tags) in enumerate(train_dataloader):
            char = torch.LongTensor(inputs[0]).to(device)
            word = torch.LongTensor(inputs[1]).to(device)
            tags = tags.to(device)
            loss = model.log_likelihood([char, word], tags)
            train_loss += loss.item()

            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                avg_loss = test_step(model, test_dataloader, id2token, id2label, save_path)
                train_loss = train_loss / len(train_dataloader.dataset)
                report = return_report(save_path)
                eval_f1 = float(report[1].strip().split()[-1])
                print(
                    "Epoch:{}, idx:{}, Train loss:{}, Dev Loss:{}, Eval F1:{}".format(epoch, idx, train_loss, avg_loss,
                                                                                      eval_f1))
                if eval_f1 > best_f1:
                    best_f1 = eval_f1
                    torch.save(model, "model/best_model.pt")


def test_step(model, test_dataloader, id2token, id2label, save_path):
    model.eval()
    with torch.no_grad():
        total_loss = 0.
        results = []
        for (char, word), tags in test_dataloader:
            # char = torch.LongTensor(inputs[0]).to(device)
            # word = torch.LongTensor(inputs[1]).to(device)
            char = char.to(device)
            word = word.to(device)
            tags = tags.to(device)
            y_pred = model(char, word).detch().cpu().numpy()
            batch_seq_len = [len([c for c in char[i] if c > 0]) for i in range(char.size()[0])]
            loss = model.log_likelihood([char, word], tags)
            total_loss += loss.item()
            for i in range(len(char)):
                result = []
                id2str = [id2token[c] for c in char[i].detach().cpu().numpy()[:batch_seq_len[i]]]
                true_tag = [id2label[t] for t in tags[i].detach().cpu().numpy()[:batch_seq_len[i]]]
                pred_tag = [id2label[t] for t in y_pred[i]]
                for c, tt, pt in zip(id2str, true_tag, pred_tag):
                    result.append(" ".join([c, tt, pt]))
                results.append(result)
        avg_loss = total_loss / len(test_dataloader.dataset)
        with open(save_path, "w", encoding="utf-8") as f:
            to_write = []
            for r in results:
                for line in r:
                    to_write.append(line + "\n")
                to_write.append("\n")
            f.writelines(to_write)

    model.train()
    return avg_loss


def main():
    # 词向量文件路径
    char_file = "/home/wangjian/workspace/ner/train/workspace/test_proj_aspect_sentiment/" \
                "embedding/test_proj_glove_char.txt"
    word_file = "/home/wangjian/workspace/ner/train/workspace/test_proj_aspect_sentiment/" \
                "embedding/test_proj_glove_word.txt"
    save_path = "result/pred_txt"

    # 加载词向量
    char_token2id, char_emb = load_embedding(char_file)
    word_token2id, word_emb = load_embedding(word_file)

    # 创建id到字的映射，预测时用到
    id2char_token = {v: k for k, v in char_token2id.items()}

    # 构建输入数据
    train_input_chars, train_inputs_words, train_labels, tags2id = load_data("../data/train/3squirrels/aspect.train",
                                                                             char_token2id, word_token2id)
    dev_input_chars, dev_inputs_words, dev_labels, _ = load_data("../data/train/3squirrels/aspect.dev",
                                                                 char_token2id, word_token2id, load_test_data=True)
    # 创建id到tag的映射，预测时用到
    id2tag = {v: k for k, v in tags2id.items()}

    # 创建模型需要的dataset
    train_dataset = NERDataset(train_input_chars, train_inputs_words, train_labels)
    dev_dataset = NERDataset(dev_input_chars, dev_inputs_words, dev_labels)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=256)

    model = NER(hidden_size=hidden_size, char_emb=char_emb, word_emb=word_emb,
                tags_num=len(tags2id), dropout_rate=dropout_rate)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, )
    train_step(model=model, epochs=epochs, train_dataloader=train_dataloader, test_dataloader=dev_dataloader,
               optimizer=optimizer, id2token=id2char_token, id2label=id2tag, save_path=save_path)


if __name__ == '__main__':
    main()
