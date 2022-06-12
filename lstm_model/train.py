from os.path import join
import torch
import torch.utils.data as data
from deep_utils import mkdir_incremental
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import GenderRecognition, collate_fn
from model import LSTMModel
from pytorch_lightning import LightningModule
from settings import Config
from torchmetrics import F1Score, Accuracy
import pytorch_lightning as pl
from deep_utils import BlocksTorch


class LitModel(LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()

        self.model = LSTMModel(classifier_output=Config.classifier_output, feature_size=Config.feature_size,
                               hidden_size=Config.hidden_size, num_layers=Config.num_layers,
                               dropout=Config.dropout, bidirectional=Config.bidirectional, device=Config.device)
        self.criterion = CrossEntropyLoss()
        self.f1_score = F1Score(num_classes=Config.num_classes, average="weighted").to(Config.device)
        self.accuracy = Accuracy(num_classes=Config.num_classes).to(Config.device)

    def forward(self, x):
        mfcc, label = x
        logit = self.model(mfcc)
        pred = torch.sigmoid(logit)
        return pred

    def get_step_metrics(self, batch):
        mfcc, labels = batch
        batch_size = mfcc.size(1)
        logits = self.model(mfcc)
        loss = self.criterion(logits, labels) * batch_size
        preds = torch.sigmoid(logits)
        corrects = torch.sum(preds == labels)
        return dict(loss=loss, corrects=corrects,
                    labels=labels, preds=preds,
                    batch_size=batch_size)

    def calculate_metrics(self, outputs, data_type="train"):
        labels, preds = [], []
        r_loss, size = 0, 0
        for row in outputs:
            r_loss += row["loss"]
            size += row["batch_size"]
            preds.append(row['preds'])
            labels.append(row["labels"])
        preds, labels = torch.concat(preds), torch.concat(labels).to(torch.long)
        f1_value = self.f1_score(preds, labels)
        acc = self.accuracy(preds, labels)
        loss = r_loss / size
        return {f"{data_type}_acc": acc, f"{data_type}_f1-score": f1_value, f"{data_type}_loss": loss}

    def training_step(self, batch, batch_idx):
        return self.get_step_metrics(batch)

    def validation_step(self, batch, batch_idx):
        return self.get_step_metrics(batch)

    def test_step(self, batch, batch_idx):
        return self.get_step_metrics(batch)

    def training_epoch_end(self, outputs) -> None:
        metrics = self.calculate_metrics(outputs, data_type="train")
        for n, m in metrics.items():
            self.log(n, m)

    def validation_epoch_end(self, outputs) -> None:
        metrics = self.calculate_metrics(outputs, data_type="val")
        for n, m in metrics.items():
            self.log(n, m)

    def test_epoch_end(self, outputs) -> None:
        metrics = self.calculate_metrics(outputs, data_type="test")
        for n, m in metrics.items():
            self.log(n, m)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=Config.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.lr_reduce_factor,
                                      patience=Config.lr_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def get_loaders():
        train_dataset = GenderRecognition(data_json=Config.train_json_path, n_classes=Config.num_classes,
                                          sample_rate=Config.sample_rate, valid=False)
        test_dataset = GenderRecognition(data_json=Config.test_json_path, n_classes=Config.num_classes,
                                         sample_rate=Config.sample_rate, valid=True)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=Config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=Config.n_workers,
                                  # drop_last=True,
                                  pin_memory=Config.pin_memory
                                  )

        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=Config.batch_size,
                                      shuffle=False,
                                      collate_fn=collate_fn,
                                      num_workers=Config.n_workers,
                                      # drop_last=True,
                                      pin_memory=Config.pin_memory
                                      )

        return train_loader, test_loader


def main():
    output_dir = mkdir_incremental(Config.output_dir)
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       filename=Config.file_name,
                                       monitor="val_loss",
                                       verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=1 if Config.device == "cuda" else 0,
                         max_epochs=Config.epochs,
                         min_epochs=Config.epochs // 10,
                         callbacks=[model_checkpoint, learning_rate_monitor],
                         default_root_dir=output_dir)
    lit_model = LitModel()
    lit_model.model.apply(BlocksTorch.weights_init)
    train_loader, val_loader = lit_model.get_loaders()
    print("[INFO] Training the model")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(lit_model, ckpt_path="best", dataloaders=val_loader)
    trainer.test(lit_model, ckpt_path="best", dataloaders=train_loader)

    weight_path = join(output_dir, f"{Config.file_name}.ckpt")
    best_weight = torch.load(weight_path)
    best_weight['config'] = Config
    torch.save(best_weight, weight_path)


if __name__ == '__main__':
    main()
