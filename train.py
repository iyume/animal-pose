from pathlib import Path
from typing import ClassVar, Optional, cast

import click
from torch import Tensor
import torch
import torch.optim
from torch import nn
from torch.utils.data import DataLoader

from datasets import AnimalPose
from model import Net
from utils import State
from keypoints import Annotation

dataset = AnimalPose(dataset_dir="/home/iyume/datasets/Animal-Pose")

ckpt_dir = Path("ckpt")
ckpt_dir.mkdir(exist_ok=True)


class Trainer:
    model: ClassVar = Net(3, 20 * 2)  # 额外增加一个类标注非关键点
    model.train()

    def __init__(
        self,
        device: str = "cpu",
        learning_rate: float = 1e-3,
        batch_size: int = 10,
        pth_file: Optional[str] = None,
    ) -> None:
        # assert batch_size == 1, "因为图片尺寸不一致，必须使用 1 batch"
        self.device = torch.device(device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        self.loss_fn = nn.MSELoss()
        if pth_file is None:
            state = State(
                epoch=0,
                model_state_dict=self.model.state_dict(),
                optim_state_dict=self.optimizer.state_dict(),
                loss=0,
                all_losses=[],
            )
        else:
            state = cast(State, torch.load(pth_file, self.device))
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optim_state_dict"])
            # get parameters view
            state["model_state_dict"] = self.model.state_dict()
            state["optim_state_dict"] = self.optimizer.state_dict()
        self.state = state

    def train_one(self, create_checkpoint: bool = True):
        loss_history = []
        images: Tensor
        keypoints: Tensor
        for idx, (images, keypoints) in enumerate(self.dataloader):
            images = images.to(self.device)
            keypoints = keypoints.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(images)
            loss: torch.Tensor = self.loss_fn(out, keypoints)
            loss.backward()
            loss_history.append(loss.item())
            self.optimizer.step()
            print("iter: {}  loss: {:.4f}".format(idx, loss))
        self.state["epoch"] += 1
        loss_avg = sum(loss_history) / len(loss_history)
        self.state["loss"] = loss_avg
        self.state["all_losses"].append(loss_avg)
        print(f"epoch {self.state['epoch']} training complete")
        print(f"STAT LOSS: {self.state['loss']:.4f}")
        if create_checkpoint:
            checkpoint = (
                ckpt_dir / f"model_{self.model.version}_epoch{self.state['epoch']}.pth"
            )
            torch.save(self.state, checkpoint)
            print(f"model saved at {checkpoint}")

    def train(self, num_epochs: int = 100):
        for _ in range(num_epochs):
            self.train_one(True)


@click.command()
@click.option(
    "-p", "--pth-file", default=None, type=click.Path(exists=True, dir_okay=False)
)
def cli(pth_file: Optional[str]):
    trainer = Trainer(pth_file=pth_file)
    trainer.train()


if __name__ == "__main__":
    Trainer().train()
