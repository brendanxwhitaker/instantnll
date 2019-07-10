from typing import List

import torch
from overrides import overrides

from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

try:
    from simrel import SimRel
except ImportError:
    from instantnll.simrel import SimRel

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
                labels: torch.Tensor,
                class_avgs: torch.Tensor, # MODIFIED.
                mask: torch.LongTensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A tensor of shape (batch_size, instance_length, input_dim)
        labels : ``torch.Tensor``, required.
            A tensor of shape (batch_size, input_dim)
        class_avgs : ``torch.Tensor``, required.
            A tensor of shape (num_classes, input_dim)
        mask : ``torch.LongTensor``, optional (default = None).
            A tensor of shape (batch_size, instance_length).
        Returns
        -------
        A tensor of shape (batch_size, seq_len, output_dim).
        Modifies
        --------
        class_avgs : the class average vectors.
        """
        if mask is None:
            return self._simrel(inputs, labels, class_avgs) # Modifies `class_avgs`.
        else:
            outputs = self._simrel(inputs, labels, class_avgs) # Modifies `class_avgs`.
            return outputs * mask.unsqueeze(dim=-1).float()
