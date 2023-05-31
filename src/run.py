import csv
import os
import pytorch_lightning as pl
from pytorch_lightning import loggers

from create_dataloader import CreateDataLoader
from model import Bert_pl


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 使用するmodelを指定
# MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
# MODEL_NAME = "cl-tohoku/bert-large-japanese"
# MODEL_NAME = "rinna/japanese-roberta-base"
MODEL_NAME = ""  # JTweetRoBERTa

# dataloaderの作成, Tokenizerのモデルも指定
cdl = CreateDataLoader(MODEL_NAME)
dataloader_train, dataloader_val, dataloader_test = cdl.create()

monitor_loss = "val_MAE"

# checkpointの設定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor=monitor_loss,
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="",  # modelの保存先
    filename="",  # 各epochの保存先
)

# early stoppingの設定
early_stopping = pl.callbacks.EarlyStopping(
    monitor=monitor_loss,
    mode="min",
    patience=5,
)

# trainerの設定
trainer = pl.Trainer(gpus=[7], max_epochs=100, callbacks=[checkpoint, early_stopping], logger=[loggers.TensorBoardLogger("")]) #logの保存先

# trainの実行
model = Bert_pl(MODEL_NAME)
trainer.fit(model, dataloader_train, dataloader_val)

# best modelの表示
best_model_path = checkpoint.best_model_path
print("Best model's file: ", checkpoint.best_model_path)

# test dataと結果の保存先の設定
with open("", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["target", "pred", "text", "follower", "party_type"])

# testの実行
test = trainer.test(dataloaders=dataloader_test, ckpt_path=checkpoint.best_model_path)

# testの結果の表示
result_dict = trainer.callback_metrics
Loss = result_dict["LOSS"]
mse = result_dict["MSE"]
rmse = result_dict["RMSE"]
mae = result_dict["MAE"]
r2 = result_dict["R2"]
corrcoef = result_dict["CorrCoef"]
mape = result_dict["MAPE"]

result = f",{Loss},{mse},{rmse},{mae},{r2},{corrcoef},{mape}"
result_list = [",LOSS,MSE,RMSE,MAE,R2,CorrCoef,MAPE"]
result_list.append(result)

print(result_list)
