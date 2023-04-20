import csv
import random
import re
from glob import glob
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.font_manager import FontProperties
from numpy import log2
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class CreateDataLoader:
    def __init__(self, model_name):
        self.model_name = model_name

    def create(self, debag=False):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # tokenizer = AutoTokenizer.from_pretrained(self.model_name, mecab_kwargs={"mecab_dic": "unidic", "mecab_option": None})

        max_length = 35
        dataset_for_loader = []
        dataset_for_loader_buzz = []
        
        retweet_list = []
        text_len_list = []

        files = glob("") #random sampling dataのpath
        
        for file in files:
            file_csv = pd.read_csv(file, encoding="utf-8-sig", sep=",")

            zero_count = 0

            for i in range(len(file_csv)):
                if debag is True:
                    if i >= 10:
                        break
                
                text = str(file_csv["text"][i])
                text = re.sub('""', 'wquote', text)
                text = re.sub('"', "", text)
                text = re.sub("wquote", '"', text)
                text = re.sub("", "", text)
                
                targets = file_csv["retweet"][i]
                
                if re.search(r"[ぁ-んァ-ン]", text) is False:
                    print("Not Japanese")
                    continue
                
                if targets < 1:
                    continue
                
                retweet_list.append(targets)
                text_len_list.append(len(text))
                
                targets = log2(targets + 1)
                    
                followers = [file_csv["followers_count"][i]]
                party_type = [file_csv["party_type"][i]]
                
                encoding = tokenizer(
                    text, max_length=max_length, padding="max_length", truncation=True, 
                )
                encoding = {key: torch.tensor(value) for key, value in encoding.items()}
                encoding["targets"] = torch.tensor(targets, dtype=torch.float)
                encoding["followers"] = torch.tensor(followers, dtype=torch.float)
                encoding["party_type"] = torch.tensor(party_type, dtype=torch.float)
                encoding["text"] = text
                dataset_for_loader.append(encoding)
                
        buzz_files = glob("") # buzz dataのpath
        
        for buzz_file in buzz_files:
            buzz_csv = pd.read_csv(buzz_file, encoding="utf-8-sig", sep=",")
            
            for i in range(len(buzz_csv)):
                if debag is True:
                    if i >= 10:
                        break
                
                text = str(buzz_csv["text"][i])
                text = re.sub('""', 'wquote', text)
                text = re.sub('"', "", text)
                text = re.sub("wquote", '"', text)
                text = re.sub("[\u3000 \t]", "", text)
                
                targets = buzz_csv["retweet"][i]
                
                if re.search(r"[ぁ-んァ-ン]", text) is False:
                    print("Not Japanese")
                    continue
                
                if targets > 300000:
                    continue
                
                if targets < 1:
                    continue
                
                retweet_list.append(targets)
                text_len_list.append(len(text))
                
                targets = log2(targets + 1)
                
                followers = [buzz_csv["followers_count"][i]]
                party_type = [buzz_csv["party_type"][i]]
                
                encoding = tokenizer(
                    text, max_length=max_length, padding="max_length", truncation=True
                )
                encoding = {key: torch.tensor(value) for key, value in encoding.items()}
                encoding["targets"] = torch.tensor(targets, dtype=torch.float)
                encoding["followers"] = torch.tensor(followers, dtype=torch.float)
                encoding["party_type"] = torch.tensor(party_type, dtype=torch.float)
                encoding["text"] = text
                dataset_for_loader_buzz.append(encoding)
        
        n = len(dataset_for_loader)
        n_train = int(0.95 * n)
        n_val = int(0.025 * n)
        
        buzz_n = len(dataset_for_loader_buzz)
        buzz_n_train = int(0.8 * buzz_n)
        buzz_n_val = int(0.1 * buzz_n)
        
        dataset_train = dataset_for_loader[:n_train] + dataset_for_loader_buzz[:buzz_n_train]
        dataset_val = dataset_for_loader[n_train : n_train + n_val] + dataset_for_loader_buzz[buzz_n_train : buzz_n_train + buzz_n_val]
        dataset_test = dataset_for_loader[n_train + n_val :] + dataset_for_loader_buzz[buzz_n_train + buzz_n_val :]
        
        print(f"The number of data : {n+buzz_n}{n , buzz_n}")
        print(f"The number of train : {len(dataset_train)}{n_train, buzz_n_train}")
        print(f"The number of valid : {len(dataset_val)}{n_val, buzz_n_val}")
        print(f"The number of test : {len(dataset_test)}{(n - n_train - n_val), (buzz_n - buzz_n_train - buzz_n_val)}")
        
        print(f"max_retweet: {max(retweet_list)}")
        print(f"min_retweet: {min(retweet_list)}")
        print(f"mean_retweet: {sum(retweet_list)/len(retweet_list)}")
        print(f"median_retweet: {median(retweet_list)}")
        print(f"max_text_len: {max(text_len_list)}")
        print(f"min_text_len: {min(text_len_list)}")
        print(f"mean_text_len: {sum(text_len_list)/len(text_len_list)}")

        plt.hist(retweet_list,bins=2**np.linspace(0,18.5,50),log=True)
        plt.xlabel("Retweet", fontsize=20)
        plt.xscale("log", base=2)
        plt.ylabel("Frequency", fontsize=20)
        plt.savefig("retweet4.png")

        dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=256)
        dataloader_test = DataLoader(dataset_test, batch_size=256)

        return dataloader_train, dataloader_val, dataloader_test

# debug
# MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
# cdl = CreateDataLoader(MODEL_NAME)
# dataloader_train, dataloader_val, dataloader_test = cdl.create()
# print("finish")
