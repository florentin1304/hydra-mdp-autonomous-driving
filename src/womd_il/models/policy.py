"""Encoder + decoder policy used by training and closed-loop eval."""

from __future__ import annotations

from torch import Tensor, nn

from womd_il.models.encoder import WayformerEncoder
from womd_il.models.interface import DecoderBase, SceneEncoding, TrajectoryBundle


class WayformerPolicy(nn.Module):
    def __init__(self, encoder: WayformerEncoder, decoder: DecoderBase):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, batch: dict[str, Tensor]) -> SceneEncoding:
        return self.encoder(batch)

    def infer(self, batch: dict[str, Tensor], **kwargs) -> TrajectoryBundle:
        return self.decoder.infer(self.encoder(batch), **kwargs)

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.decoder.train_step(self.encoder(batch), batch)

    @property
    def decoder_name(self) -> str:
        return getattr(self.decoder, "name", type(self.decoder).__name__)
