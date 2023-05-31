import csv

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import (MeanAbsoluteError, MeanAbsolutePercentageError,
                          PearsonCorrCoef, R2Score)
from transformers import AutoModel


class Bert_pl(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        # ハイパーパラメータの定義
        self.save_hyperparameters()
        # modelの定義
        self.bert = AutoModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

        # レイヤーの定義
        self.cls_layer1 = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.ff1 = nn.Linear(128, 1)

        # 損失関数,評価指標の定義
        self.mean_squared_error = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.r2_score = R2Score()
        self.corr_coef = PearsonCorrCoef()
        self.mean_absolute_percentage_error = MeanAbsolutePercentageError()

        # パラメータの凍結の設定
        for param in self.bert.parameters():
            param.requires_glad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_glad = True

    # forwardの定義,ここでモデルの順伝搬を定義する
    def forward(self, input_ids, attention_mask, followers=None, party_type=None, text=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]
        output = self.cls_layer1(logits)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.ff1(output)

        return output, outputs, text

    # train時の処理
    def training_step(self, batch, batch_idx):
        batch["targets"] = batch["targets"].reshape(batch["attention_mask"].shape[0], -1)
        output, _, text = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], followers=batch["followers"], party_type=batch["party_type"], text=batch["text"])
        target = batch["targets"]
        loss = self.mean_absolute_error(output, target)

        return {"loss": loss, "batch_preds": output, "batch_targets": target, "text": text}

    # val時の処理
    def validation_step(self, batch, batch_idx):
        batch["targets"] = batch["targets"].reshape(batch["attention_mask"].shape[0], -1)
        output, _, text = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], followers=batch["followers"], party_type=batch["party_type"], text=batch["text"])
        target = batch["targets"]
        loss = self.mean_absolute_error(output, target)

        return {"val_loss": loss, "batch_preds": output, "batch_targets": target, "text": text}

    # test時の処理
    def test_step(self, batch, batch_idx):
        batch["targets"] = batch["targets"].reshape(batch["attention_mask"].shape[0], -1)
        output, _, text = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], followers=batch["followers"], party_type=batch["party_type"], text=batch["text"])
        target = batch["targets"]
        loss = self.mean_absolute_error(output, target)

        for i in range(target.size()[0]):
            with open("", "a") as f:
                writer = csv.writer(f)
                writer.writerow([target[i].item(), output[i].item(), text[i], batch["followers"][i].item(), batch["party_type"][i].item()])

        return {"test_loss": loss, "batch_preds": output, "batch_targets": target, "text": text}

    # train epoch終了時の処理
    def training_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "train_")

    # val epoch終了時の処理
    def validation_epoch_end(self, outputs, result="val_"):
        preds = torch.cat([x["batch_preds"] for x in outputs])
        targets = torch.cat([x["batch_targets"] for x in outputs])

        MSE = self.mean_squared_error(preds, targets)
        RMSE = np.sqrt(float(MSE))
        MAE = self.mean_absolute_error(preds, targets)
        R2 = self.r2_score(preds, targets)
        CorrCoef = self.corr_coef(preds, targets)

        LOSS = preds / targets
        LOSS = LOSS.mean()
        LOSS = torch.abs(LOSS)

        MAPE = self.mean_absolute_percentage_error(preds, targets)

        metrics = {f"{result}LOSS": LOSS, f"{result}MSE": MSE, f"{result}RMSE": RMSE, f"{result}MAE": MAE, f"{result}R2": R2, f"{result}CorrCoef": CorrCoef, f"{result}MAPE": MAPE}

        self.log_dict(metrics, logger=True)

    # test epoch終了時の処理
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "")

    # optimizerの定義
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params": self.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
            {"params": self.ff1.parameters(), "lr": 1e-5}
        ])

        return [optimizer]
