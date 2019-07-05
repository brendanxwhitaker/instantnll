"""
A similarity/relatedness computation against a fixed number of class avg vectors. 
"""
from typing import List, Union
from numpy.testing import assert_almost_equal

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

    def forward(self, 
                inputs: torch.Tensor, 
                labels: torch.Tensor, 
                class_avgs: List[torch.LongTensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        assert self._output_dim == len(class_avgs)
        cosine_similarity = CosineSimilarity()
   
        """
        I want to loop over inputs vectors, so each row of inputs. Recall there are (batch_size,) rows. 
        For each row/vector, I want to compute the cosine similarity of that vector with every vector
        in class_avgs. 

        We must treat the case where we have seen none of the nontrivial classes yet, where we have
        seen only some of them, and where we have seen all of them.

        Perhaps it doesn't make much sense to train on the trivial class. Yes it doesn't make sense
        to train on the trivial class, because then, in the case where the token currently being tagged
        is farther from the trivial class avg vector than it is from the nontrivial class avg vectors, 
        it will label it as a nontrivial entity, even if it's really far away from everything. This is
        bad behavior. So maybe a SimRel threshold parameter is a better approach.  
        """ 
   
        print("===TEST===")
        # print(class_avgs[0])
        if True in torch.isinf(class_avgs[0]):
            print("Infinite")
        print("===TEST===") 
        print("Shape of inputs", inputs.shape)       

 
        output = []
        for vec in inputs:
            simvals = []
            for class_vec in class_avgs:
                if True in torch.isinf(class_vec):
                    print(vec)
                    print(cosine_similarity(vec,vec))
                    simvals.append(torch.Tensor(1.0))
                else: 
                    simvals.append(cosine_similarity(vec, class_vec))
                
                # Update `class_avgs`.
                # We need to pass the labels for this sentence to SimRel. 
                """
                for class_vec in class_avgs:
                    if True in torch.isinf(class_vec):
                        class_vec = 
                """
 
            output.append(torch.stack(simvals))
        output = torch.stack(output)    # Make torch.FloatTensor
        return output
