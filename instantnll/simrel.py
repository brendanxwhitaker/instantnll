"""
A similarity/relatedness computation against a fixed number of class avg vectors. 
"""
from typing import List, Union
from numpy.testing import assert_almost_equal

import copy
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
        self.debug = False

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, 
                inputs: torch.Tensor, # Shape: (batch_size, #_tokens_in_longest_inst, input_dim) 
                labels: torch.Tensor, # Shape: (batch_size, #_tokens_in_longest_inst) 
                class_avgs: List[torch.LongTensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        if self.debug: 
            print("===SIMREL DEBUG===")
            print("Are we training:", self.training) 
            print("===SIMREL DEBUG===") 
        if self._output_dim != len(class_avgs):
            print("===================")
            print("Parameter `num_classes` in config doesn't match the total number of classes \
                  seen in the union of the training and validation datasets. ")
            print(" Is there an example of every class? Try setting `num_classes` to " \
                  + str(len(class_avgs)) + ".")
            print("===================")
            assert self._output_dim == len(class_avgs)
        cosine_similarity = CosineSimilarity()
   
        """
        Computes the cosine similarity of each word vector with every vector in class_avgs. 

        Perhaps it doesn't make much sense to train on the trivial class. Yes it doesn't make sense
        to train on the trivial class, because then, in the case where the token currently being tagged
        is farther from the trivial class avg vector than it is from the nontrivial class avg vectors, 
        it will label it as a nontrivial entity, even if it's really far away from everything. This is
        bad behavior. So maybe a SimRel threshold parameter is a better approach.  
        """ 
  
        
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
                            simvals.append(cosine_similarity(vec,vec)) 
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
                if self.debug:
                    self._print_class_avgs(class_avgs)

                batch_out.append(torch.stack(simvals))
            output.append(torch.stack(batch_out))
        output = torch.stack(output)    # Make torch.FloatTensor
        return output

    def _print_class_avgs(self, class_avgs: List[torch.Tensor]) -> None:

        for class_avg in class_avgs:
            # print(hex(id(class_avg)), end = ' ')
            print(float(class_avg[0]), end = ' ')
        print("")
