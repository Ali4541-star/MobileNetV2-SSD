import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from speech_transforms import *
from dataset import *
from torch.nn.functional import sigmoid
from pytorch_lightning.callbacks import LearningRateFinder
import sys

class LitClassifier(pl.LightningModule):

    def __init__(self, 
        model: nn.Module, 
        model_name:str, 
        metrics: dict[str, callable], 
        loss_fn: nn.Module, 
        callbacks: list[callable],
        optimizer: torch.optim.Optimizer,
        optimizer_params: dict, 
        scheduler:dict = None,
        scheduler_params:dict = None,
        *args, 
        **kwargs
    ):
        super(LitClassifier, self).__init__(*args, **kwargs)
        self.model = model
        self.model_name = model_name
        self.callbacks: list = callbacks
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.train_results = {"preds": [], "targets": [], "id": []}
        self.val_results = {"preds": [], "targets": [], "id": []}
        self.test_results = {"preds": [], "targets": [], "id": []}
        self.df_train = pd.DataFrame(columns=["PATH", "PRED", "LABEL"])
        self.df_val = pd.DataFrame(columns=["PATH", "PRED", "LABEL"])
        self.df_test = pd.DataFrame(columns=["PATH", "PRED", "LABEL"])

    def forward(self, x):

        return self.model(x)
            
    def __common_step(self, batch):

        x, target, idx = batch
        y_preds = self(x)
        #target = target.to(torch.float)
        loss = self.loss_fn(y_preds, target)
        return loss, target, idx, y_preds

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        
        loss, target, idx, y_preds = self.__common_step(batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.train_results["preds"].append(y_preds)
        self.train_results["targets"].append(target)
        self.train_results["id"].append(idx)

        return loss

    def on_train_epoch_end(self):
        
        ep_results = dict()
        preds = torch.cat(self.train_results["preds"])
        targets = torch.cat(self.train_results["targets"])
        ep_results["loss"] = self.loss_fn(preds, targets)

        self.df_train["PRED"] = preds.tolist()
        self.df_train["TARGET"] = targets.tolist()
        self.df_train["ID"] = torch.cat(self.train_results["id"]).tolist()

        for metric in self.metrics:
            value = self.metrics[metric](preds, targets)
            self.log(f"train_{metric}", value, on_step=False, on_epoch=True)
            ep_results[f"val_{metric}"] = value

        self.train_results = {x: [] for x in self.train_results}

        return ep_results

    def validation_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        
        loss, target, idx, y_preds = self.__common_step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.val_results["preds"].append(y_preds)
        self.val_results["targets"].append(target)
        self.val_results["id"].append(idx)

        return loss

    def on_validation_epoch_end(self):

        ep_results = dict()
        preds = torch.cat(self.val_results["preds"])
        targets = torch.cat(self.val_results["targets"])
        ep_results["loss"] = self.loss_fn(preds, targets)

        self.df_val["PRED"] = preds.tolist()
        self.df_val["TARGET"] = targets.tolist()
        self.df_val["ID"] = torch.cat(self.val_results["id"]).tolist()

        for metric in self.metrics:
            value = self.metrics[metric](preds, targets)
            self.log(f"val_{metric}", value, on_step=False, on_epoch=True)
            ep_results[f"val_{metric}"] = value

        self.val_results = {x: [] for x in self.val_results}

        return ep_results


    def test_step(self, batch: torch.Tensor, batch_idx) -> torch.Tensor:
        
        loss, target, idx, y_preds = self.__common_step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.test_results["preds"].append(y_preds)
        self.test_results["targets"].append(target)
        self.test_results["id"].append(idx)

        return loss

    def on_test_epoch_end(self):
        
        ep_results = dict()
        preds = torch.cat(self.test_results["preds"])
        targets = torch.cat(self.test_results["targets"])
        ep_results["loss"] = self.loss_fn(preds, targets)

        self.df_test["PRED"] = preds.tolist()
        self.df_test["TARGET"] = targets.tolist()
        self.df_test["ID"] = torch.cat(self.test_results["id"]).tolist()

        for metric in self.metrics:
            value = self.metrics[metric](preds, targets)
            self.log(f"test_{metric}", value, on_step=False, on_epoch=True)
            ep_results[f"val_{metric}"] = value

        self.df_test.to_csv(
            os.path.join(os.getcwd(), f"{self.model_name}_test_results.csv"),
            index=False
        )

        self.test_results = {x: [] for x in self.test_results}
        return ep_results

    def on_train_end(self):

        self.df_train.to_csv(
            os.path.join(os.getcwd(), f"{self.model_name}_train_results.csv"),
            index=False
        )
        self.df_val.to_csv(
            os.path.join(os.getcwd(), f"{self.model_name}_val_results.csv"),
            index=False
        )
        

    def configure_optimizers(self):

        optimizer = self.optimizer(
            self.parameters(), 
            **self.optimizer_params
        )

        if not self.scheduler:
            return optimizer

        if "monitor" in self.scheduler_params:
            monitor = self.scheduler_params["monitor"]
            self.scheduler_params.pop("monitor")
            scheduler = self.scheduler(
                optimizer, 
                **self.scheduler_params
            ) 
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor
                }
            }

        scheduler = self.scheduler(
            optimizer, 
            **self.scheduler_params
        ) 
        
        return [optimizer], [scheduler]

    def configure_callbacks(self):

        return self.callbacks



