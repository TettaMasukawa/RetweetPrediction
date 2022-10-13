import random
from glob import glob

import pandas as pd
import torch
from numpy import log1p
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer


class CreateDataLoader:
    def __init__(self, model_name):
        self.model_name = model_name

    def create(self):

        tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)

        max_length = 35
        dataset_for_loader = []
        dataset_for_loader_buzz = []

        files = glob("/home/tmasukawa/tweet-analysis/data/tweet-data-random/formatted_v2/TweetDataAll*.csv")

        for file in files:
            file_csv = pd.read_csv(file, encoding="utf-8-sig", sep=",")

            for i in range(len(file_csv)):
                text = str(file_csv["text"][i])
                targets = file_csv["retweet"][i]
                
                if targets != 0:
                    targets = log1p(targets)
                    
                followers = [file_csv["followers_count"][i]]
                
                encoding = tokenizer(
                    text, max_length=max_length, padding="max_length", truncation=True, 
                )
                encoding = {key: torch.tensor(value) for key, value in encoding.items()}
                encoding["targets"] = torch.tensor(targets, dtype=torch.float)
                encoding["followers"] = torch.tensor(followers, dtype=torch.float)
                dataset_for_loader.append(encoding)
                
        buzz_files = glob("/home/tmasukawa/tweet-analysis/data/tweet-data-random/formatted_v2/BuzzTweetData*.csv")
        
        for buzz_file in buzz_files:
            buzz_csv = pd.read_csv(buzz_file, encoding="utf-8-sig", sep=",")
            
            for i in range(len(buzz_csv)):
                text = str(buzz_csv["text"][i])
                targets = buzz_csv["retweet"][i]
                
                if targets != 0:
                    targets = log1p(targets)
                
                followers = [buzz_csv["followers_count"][i]]
                
                encoding = tokenizer(
                    text, max_length=max_length, padding="max_length", truncation=True
                )
                encoding = {key: torch.tensor(value) for key, value in encoding.items()}
                encoding["targets"] = torch.tensor(targets, dtype=torch.float)
                encoding["followers"] = torch.tensor(followers, dtype=torch.float)
                dataset_for_loader_buzz.append(encoding)

        random.shuffle(dataset_for_loader)
        random.shuffle(dataset_for_loader_buzz)
        
        n = len(dataset_for_loader)
        n_train = int(0.7 * n)
        n_val = int(0.1 * n)
        
        buzz_n = len(dataset_for_loader_buzz)
        buzz_n_train = int(0.7 * buzz_n)
        buzz_n_val = int(0.1 * buzz_n)
        
        dataset_train = dataset_for_loader[:n_train] + dataset_for_loader_buzz[:buzz_n_train]
        dataset_val = dataset_for_loader[n_train : n_train + n_val] + dataset_for_loader_buzz[buzz_n_train : buzz_n_train + buzz_n_val]
        dataset_test = dataset_for_loader[n_train + n_val :] + dataset_for_loader_buzz[buzz_n_train + buzz_n_val :]

        # dataset_train = dataset_for_loader_buzz[:buzz_n_train]
        # dataset_val = dataset_for_loader_buzz[buzz_n_train : buzz_n_train + buzz_n_val]
        # dataset_test = dataset_for_loader_buzz[buzz_n_train + buzz_n_val :]

        dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=256)
        dataloader_test = DataLoader(dataset_test, batch_size=256)

        return dataloader_train, dataloader_val, dataloader_test

# #debug
# MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
# cdl = CreateDataLoader(MODEL_NAME)
# dataloader_train, dataloader_val, dataloader_test = cdl.create()
# print("finish")
