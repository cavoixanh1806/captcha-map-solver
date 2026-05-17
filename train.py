"""Training entry-point.

Example:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.config import load_config
from src.dataset import CaptchaDataModule
from src.model import CaptchaModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    parser.add_argument("--exp-name", default=None, help="override experiment name")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.exp_name:
        cfg["logging"]["exp_name"] = args.exp_name

    pl.seed_everything(cfg["data"]["seed"], workers=True)

    dm = CaptchaDataModule(cfg)
    model = CaptchaModel(cfg)

    ckpt_dir = Path(cfg["logging"]["ckpt_dir"]) / cfg["logging"]["exp_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:03d}-{val/seq_acc:.4f}",
            monitor="val/seq_acc",
            mode="max",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/seq_acc",
            mode="max",
            patience=cfg["solver"]["early_stopping_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger(cfg["logging"]["log_dir"], name=cfg["logging"]["exp_name"])

    trainer = pl.Trainer(
        max_epochs=cfg["solver"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision=cfg["solver"]["precision"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=False,
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
