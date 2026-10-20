"""Small neural-network models used by the CAP-Adam experiments."""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn


class MLP300(nn.Module):
    """MLP with two hidden layers of 300 units each.

    Inputs with more than two dimensions are flattened across all non-batch
    dimensions, so ``input_units`` should equal the number of features in one
    flattened example.
    """

    def __init__(self, input_units: int, output_units: int) -> None:
        super().__init__()
        if input_units <= 0:
            raise ValueError("input_units must be positive")
        if output_units <= 0:
            raise ValueError("output_units must be positive")

        self.input_units = input_units
        self.output_units = output_units
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_units, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, output_units),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return logits for a batch of inputs."""
        return self.network(inputs)


class CNN(nn.Module):
    """Small three-convolution CNN designed for CIFAR-10.

    Args:
        input_channels: Number of input channels.  Defaults to CIFAR-10 RGB.
        output_units: Number of output logits.  Defaults to 10 classes.
        channels: Output widths of the three convolutional layers.

    The default model maps an input of shape ``(B, 3, 32, 32)`` to logits of
    shape ``(B, 10)``.  ``input_channels`` and ``output_units`` remain
    configurable for closely related 32x32 datasets.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_units: int = 10,
        channels: Sequence[int] = (32, 64, 128),
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if output_units <= 0:
            raise ValueError("output_units must be positive")
        if len(channels) != 3 or any(width <= 0 for width in channels):
            raise ValueError("channels must contain three positive widths")

        channel1, channel2, channel3 = channels
        self.input_channels = input_channels
        self.output_units = output_units
        self.features = nn.Sequential(
            nn.Conv2d(
                input_channels, channel1, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channel1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channel1, channel2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channel2, channel3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(channel3 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_units),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return logits for a batch of images in NCHW format."""
        if inputs.ndim != 4 or inputs.shape[1:] != (self.input_channels, 32, 32):
            raise ValueError(
                "CNN expects input shaped "
                f"(B, {self.input_channels}, 32, 32), got {tuple(inputs.shape)}"
            )
        return self.classifier(self.features(inputs))


__all__ = ["MLP300", "CNN"]
