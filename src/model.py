import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef, R2Score
from transformers import BertModel


class Bert_pl(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()

        self.save_hyperparameters()

        self.bert = BertModel.from_pretrained(model_name)
        
        self.cls_layer1 = nn.Linear(self.bert.config.hidden_size,128)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.25)
        
        self.ff1 = nn.Linear(129, 1)
        
        self.mean_squared_error = nn.MSELoss()
        self.mean_absolute_error = MeanAbsoluteError()
        self.r2_score = R2Score()
        self.corr_coef = PearsonCorrCoef()
        
        for param in self.bert.parameters():
            param.requires_glad = False
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_glad = True
        
    def forward(self, input_ids, attention_mask, followers):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu(output)
        # output = self.dropout(output)
        output = torch.cat([output, followers], dim=1)
        output = self.ff1(output)
        
        return output

    def training_step(self, batch, batch_idx):
        batch["targets"] = batch["targets"].reshape(batch["attention_mask"].shape[0], -1)
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], followers=batch["followers"])
        target = batch["targets"]
        loss = self.mean_absolute_error(output, target)
        
        return {"loss": loss, "batch_preds": output, "batch_targets": batch["targets"]}

    def validation_step(self, batch, batch_idx):
        batch["targets"] = batch["targets"].reshape(batch["attention_mask"].shape[0], -1)
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], followers=batch["followers"])
        target = batch["targets"]
        loss = self.mean_absolute_error(output, target)
        
        return {"val_loss": loss, "batch_preds": output, "batch_targets": batch["targets"]}

    def test_step(self, batch, batch_idx):
        batch["targets"] = batch["targets"].reshape(batch["attention_mask"].shape[0], -1)
        output = self.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], followers=batch["followers"])
        target = batch["targets"]
        loss = self.mean_absolute_error(output, target)
        
        return {"test_loss": loss, "batch_preds": output, "batch_targets": batch["targets"]}

    def validation_epoch_end(self, outputs, mode="val_"):
        preds = torch.cat([x["batch_preds"] for x in outputs])
        targets = torch.cat([x["batch_targets"] for x in outputs])
        MSE = self.mean_squared_error(preds, targets)
        RMSE = np.sqrt(float(MSE))
        MAE = self.mean_absolute_error(preds, targets)
        R2 = self.r2_score(preds, targets)
        CorrCoef = self.corr_coef(preds, targets)
        
        metrics = {f"{mode}MSE": MSE, f"{mode}RMSE": RMSE, f"{mode}MAE": MAE, f"{mode}R2": R2, f"{mode}CorrCoef": CorrCoef}
        
        self.log_dict(metrics, logger=True)
        
    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, "")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {"params":self.bert.encoder.layer[-1].parameters(), "lr":5e-5},
            {"params":self.ff1.parameters(), "lr":1e-5}
        ])
        
        return [optimizer]
