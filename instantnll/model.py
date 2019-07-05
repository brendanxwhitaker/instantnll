from typing import Iterator, List, Dict

import os
import torch
import torch.optim as optim
import numpy as np

import shutil
import tempfile

from allennlp_debug.train import train_model
from allennlp.common.params import Params
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

from dataset_reader import InstDatasetReader
from encoder import CosineEncoder
from allennlp_debug.text_field_embedder import BasicTextFieldEmbedder
from allennlp_debug.embedding import Embedding
torch.manual_seed(1)

@Model.register('instantnll')
class EntityTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.label_vocab = vocab.get_index_to_token_vocabulary(namespace='labels')

        inf_vec = torch.Tensor([float('-inf')] * encoder.get_input_dim())
        self.class_avgs = [inf_vec.clone() for i in range(len(self.label_vocab))]

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()

        print("===DEBUG===")
        print(vocab)
        print(vocab.get_index_to_token_vocabulary(namespace='labels'))
        print("===DEBUG===")

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        print("===DEBUG===")
        print(labels)
        print("Sentence:", sentence)
        print("===DEBUG===")

        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        print("===DEBUG===")
        print("Shape of embeddings", embeddings.shape) 
        print("===DEBUG===") 
        class_avgs = self.class_avgs
        encoder_out = self.encoder(embeddings, labels, class_avgs, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

if __name__ == '__main__':
    params = Params.from_file('experiment.jsonnet')
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    # Make predictions
    predictor = SentenceTaggerPredictor(model, dataset_reader=InstDatasetReader())
    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    print(tag_logits)
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    shutil.rmtree(serialization_dir)
