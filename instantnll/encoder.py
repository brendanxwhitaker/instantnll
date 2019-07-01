import torch
from overrides import overrides

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

@Seq2SeqEncoder.register("CosineEncoder")
class CosineEncoder(Seq2SeqEncoder):
    """
    This class applies the `SimRel` to each item in sequences.
    """
    def __init__(self, simrel: SimRel) -> None:
        super().__init__()
        self._simrel = simrel

    @overrides
    def get_input_dim(self) -> int:
        return self._simrel.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        return self._simrel.get_output_dim()

    @overrides
    def is_bidirectional(self) -> bool:
        return False

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                mask: torch.LongTensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, timesteps, input_dim)
        mask : ``torch.LongTensor``, optional (default = None).
            A tensor of shape (batch_size, timesteps).
        Returns
        -------
        A tensor of shape (batch_size, timesteps, output_dim).
        """
        if mask is None:
            return self._simrel(inputs)
        else:
            outputs = self._simrel(inputs)
            return outputs * mask.unsqueeze(dim=-1).float()
