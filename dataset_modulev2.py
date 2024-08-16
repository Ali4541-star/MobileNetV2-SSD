import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from speech_transforms import *
from os import path
import os
import sys
from dataclasses import dataclass






class SoundDatasetModule(pl.LightningDataModule):

    TRAIN_IDX = 0
    DEV_IDX = 1
    EVAL_IDX = 2

    def __init__(
        self, 
        datafile_paths: list[str], 
        batch_size: int, 
        fn_extract: FeatureExtractor, 
        device:str = "cpu",
        apply_batch: bool = False,
    ):
        super(SoundDatasetModule, self).__init__()
        self.datafile_paths = datafile_paths
        self.batch_size = batch_size
        self.fn_extract = fn_extract
        self.device = device
        self.apply_batch = apply_batch

    def setup(self, stage: str):
        self.train_dataset = SoundDataset(
            self.datafile_paths[self.TRAIN_IDX],
            fn_extract=self.fn_extract,
            device=self.device,
        )
        self.val_dataset = SoundDataset(
            self.datafile_paths[self.DEV_IDX],
            fn_extract=self.fn_extract,
            device=self.device,
        )
        self.test_dataset = SoundDataset(
            self.datafile_paths[self.EVAL_IDX],
            fn_extract=self.fn_extract,
            device=self.device,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
        )

    def val_dataloader(self):
        
        return DataLoader(
            self.val_dataset, 
            batch_size=len(self.val_dataset) if not self.apply_batch else self.batch_size, 
            shuffle=True, 
        )

    def test_dataloader(self):
        
        return DataLoader(
            self.test_dataset, 
            batch_size=len(self.val_dataset) if not self.apply_batch else self.batch_size, 
            shuffle=True, 
        )

    

class SoundDataset(Dataset):

    def __init__(
        self,
        datafile_path: str, 
        fn_extract: FeatureExtractor, 
        device:str = "cpu", 
    ):
        self.datafile: pd.DataFrame = pd.read_csv(datafile_path, sep=",")
        self.fn_extract = fn_extract
        self.device = device

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, pd.core.series.Series]:
        
        return (
            self.fn_extract(self.datafile.iloc[idx]["PATH"]),#.to(self.device),
            torch.tensor(self.datafile.iloc[idx]["LABEL"]),
            torch.tensor(self.datafile.iloc[idx]["ID"])
        )

    def get_dim(self) -> tuple[int, int]:  # n_samples, n_class

        return len(self), len(self["LABEL"].nunique())

    def __len__(self):
        return len(self.datafile)



class CacheDatasetModule(pl.LightningDataModule):

    TRAIN_IDX = 0
    DEV_IDX = 1
    EVAL_IDX = 2

    def __init__(
        self,
        datafile_paths: str,
        caches: list[torch.Tensor],
        batch_size: int,
        device:str = "cpu",
        apply_batch: bool = False,
    ):
        
        super(CacheDatasetModule, self).__init__()
        self.datafile_paths = datafile_paths
        self.caches = caches
        self.batch_size = batch_size
        self.device = device
        self.apply_batch = apply_batch


    def setup(self, stage: str):
        self.train_dataset = CacheSoundDataset(
            datafile_path=self.datafile_paths[self.TRAIN_IDX],
            device=self.device,
            cache=self.caches[self.TRAIN_IDX],
        )
        self.val_dataset = CacheSoundDataset(
            datafile_path=self.datafile_paths[self.DEV_IDX],
            device=self.device,
            cache=self.caches[self.DEV_IDX],
        )
        self.test_dataset = CacheSoundDataset(
            datafile_path=self.datafile_paths[self.EVAL_IDX],
            device=self.device,
            cache=self.caches[self.EVAL_IDX],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
        )

    def val_dataloader(self):
        
        return DataLoader(
            self.val_dataset, 
            batch_size=len(self.val_dataset) if not self.apply_batch else self.batch_size, 
            shuffle=True, 
        )

    def test_dataloader(self):
        
        return DataLoader(
            self.test_dataset, 
            batch_size=len(self.test_dataset) if not self.apply_batch else self.batch_size, 
            shuffle=True, 
        )



class CacheSoundDataset(Dataset):

    def __init__(
        self,
        datafile_path: str,
        cache: torch.Tensor|None,
        device:str = "cpu", 
    ):
        self.datafile: pd.DataFrame = pd.read_csv(datafile_path, sep=",")
        self.device = device
        self.cache = cache

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, pd.core.series.Series]:
        
        label: torch.Tensor = torch.zeros(self.get_dim()[1])
        label[self.datafile["LABEL"].iloc[idx]] = 1
        return (
            self.cache[idx],
            label,
            torch.tensor(self.datafile.iloc[idx]["ID"])
        )

    def get_dim(self) -> tuple[int, int]:  # n_samples, n_class

        return self.cache.shape[0], self.datafile["LABEL"].nunique()

    def __len__(self):
        return len(self.datafile)




    
    


    
