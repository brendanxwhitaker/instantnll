"""
A similarity/relatedness computation against a fixed number of class avg vectors. 
"""
from typing import List, Union

import torch

from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.nn import Activation
from allennlp.modules.similarity_functions import CosineSimilarity


class SimRel(torch.nn.Module, FromParams):
    """
    This ``Module`` applies a simple similarity/relatedness computation to each input vector
    against each of a fixed number of class avg vectors. 

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of the input.  We assume the input has shape ``(batch_size, input_dim)``.
    class_avgs : ``List[torch.LongTensor]``
        The average of the embedding vectors for each seen class. Has shape (num_classes,).
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

    def forward(self, inputs: torch.Tensor, class_avgs: List[torch.LongTensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        assert self._output_dim == len(class_avgs)
        cos_sim = CosineSimilarity()
   
        """
        I want to loop over inputs vectors, so each row of inputs. Recall there are (batch_size,) rows. 
        For each row/vector, I want to compute the cosine similarity of that vector with every vector
        in class_avgs. 
        """ 
        
        output = []
        for vec in inputs:
            simvals = []
            for class_vec in class_avgs:
                simvals.append(cos_sim(vec, class_vec))
            output.append(torch.stack(simvals))
        output = torch.stack(output)
        return output
