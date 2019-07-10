"""
A similarity/relatedness computation against a fixed number of class avg vectors.
"""
from typing import List

import torch

from allennlp.common import FromParams
from allennlp.modules.similarity_functions import CosineSimilarity

class SimRel(torch.nn.Module, FromParams):
    """
    This ``Module`` applies a simple similarity/relatedness computation to each input vector
    against each of a fixed number of class avg vectors.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    num_classes : ``int``
        The number of classes seen in the data, including an `other` class for non-relevant words.
    """
    def __init__(self, input_dim: int,
                 num_classes: int) -> None:

        super(SimRel, self).__init__()
        self.input_dim = input_dim
        self._output_dim = num_classes

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self,
                inputs: torch.Tensor,               # Shape: (batch_size, seq_len, input_dim)
                labels: torch.Tensor,               # Shape: (batch_size, seq_len)
                class_avgs: List[torch.LongTensor]  # Shape: (num_classes, input_dim)
    ) -> torch.Tensor:                              # Shape: (batch_size, seq_len, num_classes)
        # pylint: disable=arguments-differ
        """
        Computes the cosine similarity of each word vector with every vector in class_avgs.
        """
        cosine_similarity = CosineSimilarity()
        class_counts = [0] * len(class_avgs)

        output = []
        for i, batch in enumerate(inputs):
            batch_out = []
            for j, vec in enumerate(batch):
                simvals = []
                for k, class_vec in enumerate(class_avgs):

                    # If we haven't initialized a class vector.
                    if True in torch.isinf(class_vec):

                        # If we need to initialize a new class vector.
                        if labels is not None and labels[i][j] == k:
                            simvals.append(cosine_similarity(vec, vec))
                            if self.training:
                                class_avgs[k] = vec
                                class_counts[k] += 1

                        # Otherwise, we set the similarity value to -1.
                        else:
                            simvals.append(torch.tensor(-1.0))
                    else:
                        simvals.append(cosine_similarity(vec, class_vec))
                        if self.training and labels[i][j] == k:
                            class_vec_multiple = class_vec * class_counts[k]
                            class_avgs[k] = (class_vec_multiple + vec) / (class_counts[k] + 1)
                            class_counts[k] += 1

                batch_out.append(torch.stack(simvals))
            output.append(torch.stack(batch_out))
        output = torch.stack(output)    # Make torch.FloatTensor
        return output # Shape: (batch_size, sequence_length, num_classes)

