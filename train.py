from __future__ import print_function
import argparse
from pathlib import Path
from typing import Tuple, Dict
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.LPR_model as model

from models.config import PlateConfig
from models.dataset import create_dataloader
from models.utils import Averager, StringLabelConverter


class Trainer:
    def __init__(self, config: PlateConfig):
        self.config = config
        self.device = torch.device(
            f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.setup_experiment_dir()
        self.setup_model_and_data()
        self.setup_training()
        self.writer = SummaryWriter(log_dir=self.experiment_dir / "tensorboard")

    def setup_experiment_dir(self):
        self.experiment_dir = Path(self.config.experiment)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(Path(self.experiment_dir) / "config.yaml")

    def setup_model_and_data(self):
        self.train_loader = create_dataloader(
            root_path=self.config.train_data,
            batch_size=self.config.batch_size,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            enable_augmentation=True,
            enable_degrade=True,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        self.val_loader = create_dataloader(
            root_path=self.config.val_data,
            batch_size=self.config.batch_size,
            image_width=self.config.image_width,
            image_height=self.config.image_height,
            enable_augmentation=False,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        # Model setup
        self.converter = StringLabelConverter(self.config.alphabet)
        nclass = len(self.config.alphabet) + 1

        if self.config.model_type == "CNN":
            self.model = model.LPR_model(
                self.config.num_channels,
                nclass,
                imgW=self.config.image_width,
                imgH=self.config.image_height,
                K=self.config.K,
            ).to(self.device)
        else:
            raise NotImplementedError(
                f"Model type {self.config.model_type} not implemented"
            )

        if self.config.pre_model:
            print(f"Loading pretrained model from {self.config.pre_model}")
            state_dict = torch.load(self.config.pre_model, map_location=self.device)
            self.model.load_state_dict(state_dict)

    def setup_training(self):
        self.criterion = nn.CrossEntropyLoss()
        # self.loss_avg = Averager() # 不参与back
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )

        total_steps = self.config.epochs * len(self.train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.075,
        )

    def save_checkpoint(
        self, epoch: int, accuracy: float, is_best: bool = False
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "accuracy": accuracy,
        }

        # Save latest checkpoint
        torch.save(checkpoint, self.experiment_dir / "latest.pth")

        # Save best model
        if is_best:
            best_path = self.experiment_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

    @torch.no_grad()
    def validate(self, epoch: int, max_batches: int = 1000) -> Tuple[float, Dict]:
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Validation Epoch {epoch}",
            total=min(max_batches, len(self.val_loader)),
        )

        for batch_idx, (images, labels) in enumerate(pbar):
            if batch_idx >= max_batches:
                break

            images = images.to(self.device)
            batch_size = images.size(0)

            # Forward pass
            predictions = self.model(images)
            text = self.converter.encode_list(labels, self.config.K).to(self.device)

            # Calculate loss
            loss = 0
            preds_chunks = torch.chunk(predictions, predictions.size(1), 1)
            for i, pred_chunk in enumerate(preds_chunks):
                pred_chunk = pred_chunk.squeeze()
                gt = text[:, i]
                loss += self.criterion(pred_chunk, gt) / batch_size

            val_loss += loss.item()

            # Calculate accuracy
            _, pred_indices = predictions.max(2)
            pred_labels = self.converter.decode_list(pred_indices.data)

            for pred, target in zip(pred_labels, labels):
                pred = pred.replace("-", "")
                if pred == target:
                    correct += 1
            total += batch_size

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{val_loss/(batch_idx+1):.4f}",
                    "acc": f"{100.*correct/total:.2f}%",
                }
            )

        accuracy = correct / total
        metrics = {"val_loss": val_loss / (batch_idx + 1), "val_accuracy": accuracy}

        return accuracy, metrics

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            text = self.converter.encode_list(labels, self.config.K).to(self.device)

            predictions = self.model(images)
            loss = 0
            preds_chunks = torch.chunk(predictions, predictions.size(1), 1)

            for i, pred_chunk in enumerate(preds_chunks):
                pred_chunk = pred_chunk.squeeze()
                gt = text[:, i]
                loss += self.criterion(pred_chunk, gt)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            current_lr = self.scheduler.get_last_lr()[0]

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{total_loss/(batch_idx+1):.4f}", "lr": f"{current_lr:.6f}"}
            )

        metrics = {
            "train_loss": total_loss / (batch_idx + 1),
            "learning_rate": current_lr,
        }

        return metrics

    def train(self):
        best_accuracy = 0

        for epoch in range(self.config.epochs):
            # Training phase
            train_metrics = self.train_epoch(epoch)

            # Validation phase
            accuracy, val_metrics = self.validate(
                epoch, max_batches=self.config.val_max_batchs
            )

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            for name, value in all_metrics.items():
                self.writer.add_scalar(name, value, epoch)

            # Save checkpoints
            is_best = accuracy > best_accuracy
            if is_best:
                best_accuracy = accuracy

            self.save_checkpoint(epoch, accuracy, is_best)

            print(f"Epoch {epoch}: Accuracy={accuracy:.4f} (Best={best_accuracy:.4f})")
        shutil.copy(
            self.experiment_dir / "best_model.pth",
            self.experiment_dir / f"best_{best_accuracy:.4f}.pth",
        )


def main():
    parser = argparse.ArgumentParser(description="LPR Model Training Parameters")
    parser.add_argument("-p", "--params_path", type=str, default="configs/plate.yaml")
    args = parser.parse_args()
    config = PlateConfig.from_yaml(args.params_path)
    # Set random seed for reproducibility
    if hasattr(config, "seed"):
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)

    cudnn.benchmark = True

    # Start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
