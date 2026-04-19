import torch
from torch import nn


def _build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"不支持的 activation: {name}")


class MLP(nn.Module):
    """可配置的多层感知机，保持 BP/MLP 主线不变。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        weight_init: str = "kaiming",
    ):
        super().__init__()

        if hidden_dims is None:
            if hidden_dim is None:
                raise ValueError("hidden_dim 和 hidden_dims 不能同时为空。")
            hidden_dims = [hidden_dim]

        layer_dims = [input_dim] + list(hidden_dims)
        layers = []

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(_build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(layer_dims[-1], num_classes))
        self.network = nn.Sequential(*layers)

        self.activation_name = activation
        self.weight_init = weight_init
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_init == "kaiming":
                    nonlinearity = "relu" if self.activation_name == "relu" else "linear"
                    nn.init.kaiming_normal_(module.weight, nonlinearity=nonlinearity)
                elif self.weight_init == "xavier":
                    gain = nn.init.calculate_gain(self.activation_name)
                    nn.init.xavier_normal_(module.weight, gain=gain)
                else:
                    raise ValueError(f"不支持的 weight_init: {self.weight_init}")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
