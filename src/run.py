import pytorch_lightning as pl
from pytorch_lightning import loggers

from create_dataloader import CreateDataLoader
from model import Bert_pl

MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"

cdl = CreateDataLoader(MODEL_NAME)

dataloader_train, dataloader_val, dataloader_test = cdl.create()

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_MAE",
    mode="min",
    save_top_k=1,
    save_weights_only=True,
    dirpath="/disk/ssd14tc/tmasukawa/tweet-analysis/Retweet_Prediction/model",
)

trainer = pl.Trainer(gpus=[3], max_epochs=10, callbacks=[checkpoint], logger=[loggers.TensorBoardLogger("/disk/ssd14tc/tmasukawa/tweet-analysis/Retweet_Prediction/logs")])

model = Bert_pl(MODEL_NAME)

trainer.fit(model, dataloader_train, dataloader_val)

best_model_path = checkpoint.best_model_path
print("Best model's file: ", checkpoint.best_model_path)
print("loss:", checkpoint.best_model_score)

test = trainer.test(dataloaders=dataloader_test)
