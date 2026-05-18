"""Training entry-point.

Example:
    python train.py --config configs/default.yaml
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

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


def build_model(cfg):
    task = cfg["solver"].get("task", "head").lower()
    if task == "ctc":
        from src.crnn import CTCCaptchaModel

        return CTCCaptchaModel(cfg)
    if task == "trocr":
        from src.trocr import TrOCRLitModel

        return TrOCRLitModel(cfg)
    if task == "encoder_clf":
        from src.encoder_clf import EncoderClf

        return EncoderClf(cfg)
    return CaptchaModel(cfg)


def build_datamodule(cfg, model):
    task = cfg["solver"].get("task", "head").lower()
    if task == "trocr":
        from src.trocr import TrOCRDataModule

        return TrOCRDataModule(cfg, model.processor)
    if task == "encoder_clf":
        from src.encoder_clf import EncoderClfDataModule

        return EncoderClfDataModule(cfg, model.processor)
    return CaptchaDataModule(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    parser.add_argument("--exp-name", default=None, help="override experiment name")
    parser.add_argument(
        "--synth", type=int, default=None, metavar="N",
        help="override solver.synth_per_epoch: mix N synthetic images per epoch"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.exp_name:
        cfg["logging"]["exp_name"] = args.exp_name
    if args.synth is not None:
        cfg["solver"]["synth_per_epoch"] = args.synth

    # honour the Tensor Cores hint for RTX 3060+
    torch.set_float32_matmul_precision("high")

    pl.seed_everything(cfg["data"]["seed"], workers=True)

    model = build_model(cfg)
    dm = build_datamodule(cfg, model)

    ckpt_dir = Path(cfg["logging"]["ckpt_dir"]) / cfg["logging"]["exp_name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-epoch{epoch:03d}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
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
        accumulate_grad_batches=cfg["solver"].get("grad_accumulation", 1),
    )

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
