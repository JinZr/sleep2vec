import torch
import torch.nn as nn

from .base import TemporalAggregator


class LSTMAggregator(TemporalAggregator):
    name = "lstm"

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        if type(bidirectional) is not bool:
            raise ValueError("LSTM temporal aggregator bidirectional must be a boolean.")
        self.bidirectional = bidirectional

        directions = 2 if self.bidirectional else 1
        if self.hidden_size % directions != 0:
            raise ValueError("bidirectional LSTM temporal aggregator requires an even hidden_size.")

        recurrent_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size // directions,
            num_layers=self.num_layers,
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(torch.bool)
        lengths = mask.sum(dim=1).to(torch.long)
        packed = nn.utils.rnn.pack_padded_sequence(hidden, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=hidden.size(1))
        output = output * mask.unsqueeze(-1)
        denom = lengths.clamp(min=1).unsqueeze(-1).to(output.dtype)
        return output.sum(dim=1) / denom


__all__ = ["LSTMAggregator"]
