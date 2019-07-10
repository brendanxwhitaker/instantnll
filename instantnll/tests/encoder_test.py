# pylint: disable=no-self-use,invalid-name,protected-access
from numpy.testing import assert_almost_equal
from scipy.spatial.distance import cosine
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, Initializer, Activation
from allennlp.common.testing import AllenNlpTestCase

from instantnll import CosineEncoder

class TestSimRel(AllenNlpTestCase):
    def test_can_construct_from_params(self):
        params = Params({
                'input_dim': 2,
                'num_classes': 3,
                })
        simrel = SimRel.from_params(params)
        assert simrel.get_output_dim() == 3
        assert simrel.get_input_dim() == 2

    def test_forward_gives_correct_output(self):
        params = Params({
                'input_dim': 2,
                'num_classes': 1,
                })
        simrel = SimRel.from_params(params)
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(simrel)
        input_tensor = torch.FloatTensor([[[-3, 1]]])
        labels = torch.Tensor([[0]])
        class_avgs = [torch.FloatTensor([5, 5])]
        output = simrel(input_tensor, labels, class_avgs).data.numpy()
        assert output.shape == (1, 1, 1)
        # This output was checked by hand - 
        assert_almost_equal(output, torch.FloatTensor([[[-0.44721356]]]))

         
        params = Params({
                'input_dim': 5,
                'num_classes': 3,
                })
        simrel = SimRel.from_params(params)
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(simrel)
        input_tensor = torch.FloatTensor([[[1,2,3,4,5]]])
        print("Input tensor:", input_tensor)
        print("Input shape:", input_tensor.shape)
        labels = torch.Tensor([[0, 1, 2]])
        class_avgs = [torch.FloatTensor([3, 4, 5, 6, 7]), torch.FloatTensor([25,63,55,8,2.4]), torch.FloatTensor([1.003,1.005,6.578,3.4, 9.999])]
        output = simrel(input_tensor, labels, class_avgs).data.numpy()
        assert output.shape == (1, 1, 3)
        # This output was checked via WolframAlpha 
        assert_almost_equal(output, torch.FloatTensor([[[0.9864400504, 0.5535960766, 0.9296758768]]])) 
