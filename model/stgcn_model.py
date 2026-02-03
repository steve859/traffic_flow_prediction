from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """Temporal Convolution: conv over time dimension."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, F) -> (B, F, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = F.relu(x)
        # back to (B, T, N, F)
        x = x.permute(0, 2, 3, 1)
        return x


class GraphConv(nn.Module):
    """Simple Graph Convolution Layer.

    X: (B, T, N, F)
    A: (N, N)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Graph propagation: A @ X
        x = torch.einsum("ij,btjf->btif", A, x)
        # Linear transform
        x = self.linear(x)
        return x


class STGCNBlock(nn.Module):
    """STGCN Block: TemporalConv -> GraphConv -> TemporalConv."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.temp1 = TemporalConv(in_channels, hidden_channels)
        self.graph = GraphConv(hidden_channels, hidden_channels)
        self.temp2 = TemporalConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.temp1(x)
        x = self.graph(x, A)
        x = self.temp2(x)
        return x


class STGCN(nn.Module):
    """Full STGCN model matching the training notebook."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        hidden_channels: int = 32,
        out_channels: int = 64,
        num_blocks: int = 2,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        self.blocks = nn.ModuleList()
        self.blocks.append(STGCNBlock(in_channels, hidden_channels, out_channels))

        for _ in range(num_blocks - 1):
            self.blocks.append(STGCNBlock(out_channels, hidden_channels, out_channels))

        self.output = nn.Linear(out_channels, horizon)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, C)
        for block in self.blocks:
            x = block(x, A_norm)

        x = x[:, -1, :, :]  # (B, N, C)
        x = self.output(x)  # (B, N, horizon)
        return x


def normalize_adj(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5, where=d > 0)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = np.diag(d_inv_sqrt)
    return D @ A @ D


def load_adj_tensor(adj_path: str, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
    A = np.load(adj_path)
    A_norm = normalize_adj(A)
    return torch.tensor(A_norm, dtype=torch.float32, device=device)


def load_stgcn_state_dict(model_path: str, device: Union[str, torch.device] = "cpu") -> dict:
    obj = torch.load(model_path, map_location=device)
    # Some checkpoints are saved as raw state_dict; others as dict wrappers.
    if isinstance(obj, dict) and any(k.endswith("weight") or k.endswith("bias") for k in obj.keys()):
        return obj
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    raise ValueError(f"Unrecognized checkpoint format: {type(obj)}")
