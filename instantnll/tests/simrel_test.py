# pylint: disable=no-self-use,invalid-name,protected-access
from numpy.testing import assert_almost_equal
import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.nn import InitializerApplicator, Initializer, Activation
from allennlp.common.testing import AllenNlpTestCase

from instantnll import SimRel

class TestSimRel(AllenNlpTestCase):
    def test_can_construct_from_params(self):
        params = Params({
                'input_dim': 2,
                'num_classes': 3,
                })
        simrel = SimRel.from_params(params)
        assert len(simrel.get_output_dim) == 3
        assert len(simrel.get_input_dim) == 2

    def test_forward_gives_correct_output(self):
        params = Params({
                'input_dim': 2,
                'num_classes': 3,
                })
        simrel = SimRel.from_params(params)

        constant_init = Initializer.from_params(Params({"type": "constant", "val": 1.}))
        initializer = InitializerApplicator([(".*", constant_init)])
        initializer(feedforward)

        input_tensor = torch.FloatTensor([[-3, 1]])
        output = feedforward(input_tensor).data.numpy()
        assert output.shape == (1, 3)
        # This output was checked by hand - ReLU makes output after first hidden layer [0, 0, 0],
        # which then gets a bias added in the second layer to be [1, 1, 1].
        assert_almost_equal(output, [[1, 1, 1]])
