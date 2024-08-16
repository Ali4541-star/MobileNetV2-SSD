import os
from dataset_modulev2 import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from model_module import LitClassifier
import torch.nn.functional as f
import torch    
import metrics_module as mt
from torchvision.models import mobilenet_v2
from model_init import initialize_weights

def main():

    train_datafile_path = os.path.join(os.path.dirname(__file__), "dataset", "train.csv") 
    dev_datafile_path = os.path.join(os.path.dirname(__file__), "dataset", "dev.csv")
    eval_datafile_path = os.path.join(os.path.dirname(__file__), "dataset", "eval.csv")
    model_ckpt_path = os.path.join("MobileNetV2", f"BS{1}_epoch={500}")
    mobilenetv2_save_dir = os.path.join("MobileNetV2\TESTS")
    model_name = "MobileNetV2"
    lr = 0.001
    optimizer = torch.optim.Adam
    loss_fn = f.binary_cross_entropy

    mfcc_cache_paths = {
        "train": os.path.join("dataset", "cache_mfcc_train_26_coeff_3secs.pth"),
        "dev": os.path.join("dataset", "cache_mfcc_dev_26_coeff_3secs.pth"),
        "eval": os.path.join("dataset", "cache_mfcc_eval_26_coeff_3secs.pth")
    } 

    mfcc_caches = [torch.load(cache_path) for cache_path in mfcc_cache_paths.values()]

    ds_pl = CacheDatasetModule(
        datafile_paths=[train_datafile_path, dev_datafile_path, eval_datafile_path],
        caches=mfcc_caches,
        batch_size=1,
        device="cuda:0",
    )

    checkpoint = ModelCheckpoint(
        dirpath=model_ckpt_path,  # Directory to save checkpoints
        filename='_best_{epoch:1d}',
        save_top_k=10,
        monitor='val_loss'
    )

    logger=TensorBoardLogger(save_dir=mobilenetv2_save_dir, name=f"BS{1}/epoch={500}")
    
    model_mobilenetv2 = configure_mobilenetv2()

    metrics = {
        "EER": mt.EER(pos_label=0)
    }

    clf = LitClassifier(
        model=model_mobilenetv2,
        model_name=model_name,
        loss_fn=loss_fn,
        metrics=metrics,
        callbacks=[checkpoint],
        optimizer=optimizer,
        optimizer_params={"lr": lr},
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=500,
        devices=1,
        accelerator="gpu",
        num_sanity_val_steps=0,
        log_every_n_steps=1_000_000_000,
        accumulate_grad_batches=1
    )

    trainer.fit(clf, datamodule=ds_pl)
    trainer.test(clf, datamodule=ds_pl, ckpt_path=checkpoint.best_model_path)


def configure_mobilenetv2():

    mobilenetv2_model = mobilenet_v2(pretrained = False)
    mobilenetv2_model.features[0] = nn.Conv2d(1, 32, 3, 2, 1)
    mobilenetv2_model.classifier[1] = nn.Linear(1_280, 2)
    mobilenetv2_model.classifier.append(nn.Softmax())
    mobilenetv2_model.apply(initialize_weights)
    return mobilenetv2_model

if __name__ == '__main__':

    main()