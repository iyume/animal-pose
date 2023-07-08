import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import numpy as np

import torch


losses = torch.load("ckpt/model_v0.1_epoch700.pth", "cpu")["all_losses"]

assert len(losses) == 700

fig, ax = plt.subplots()

ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.plot(np.arange(700), losses)

fig.tight_layout()
plt.savefig("loss-graph.png", dpi=200)
