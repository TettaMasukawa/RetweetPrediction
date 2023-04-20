import csv
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import pytorch_lightning as pl
from pytorch_lightning import loggers

from create_dataloader import CreateDataLoader
from model import Bert_pl

# MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
# MODEL_NAME = "cl-tohoku/bert-large-japanese"
# MODEL_NAME = "rinna/japanese-roberta-base"
MODEL_NAME = "" #JTweetRoBERTa

cdl = CreateDataLoader(MODEL_NAME)

dataloader_train, dataloader_val, dataloader_test = cdl.create()

result_list = [",LOSS,MSE,RMSE,MAE,R2,CorrCoef,MAPE"]

monitor_loss = "val_MAE"

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor=monitor_loss,
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="", #modelの保存先
    filename=f"", #各epochの保存先
)

early_stopping = pl.callbacks.EarlyStopping(
    monitor=monitor_loss,
    mode="min",
    patience=5,
)

trainer = pl.Trainer(gpus=[7], max_epochs=100, callbacks=[checkpoint, early_stopping], logger=[loggers.TensorBoardLogger("")]) #logの保存先

model = Bert_pl(MODEL_NAME)

trainer.fit(model, dataloader_train, dataloader_val)

best_model_path = checkpoint.best_model_path
print("Best model's file: ", checkpoint.best_model_path)

with open(f"", "w") as f: #testの保存先
    writer = csv.writer(f)
    writer.writerow(["target","pred","text","follower","party_type"])

test = trainer.test(dataloaders=dataloader_test, ckpt_path=checkpoint.best_model_path)

result_dict = trainer.callback_metrics
Loss = result_dict["LOSS"]
mse = result_dict["MSE"]
rmse = result_dict["RMSE"]
mae = result_dict["MAE"]
r2 = result_dict["R2"]
corrcoef = result_dict["CorrCoef"]
mape = result_dict["MAPE"]

result = f",{Loss},{mse},{rmse},{mae},{r2},{corrcoef},{mape}"
result_list.append(result)

print(result_list)