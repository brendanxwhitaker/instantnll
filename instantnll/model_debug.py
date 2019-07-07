from typing import Dict

import shutil
import tempfile

import torch
import numpy as np


# from allennlp.commands.train import train_model
from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.predictors import SentenceTaggerPredictor

from dataset_reader import InstDatasetReader
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
        self.debug = False

        if self.debug:
            print("===MODEL DEBUG===")
            # print("Number of embeddings:", self.word_embeddings._token_embedders['tokens'].num_embeddings)
            # print("Embedding weights", self.word_embeddings._token_embedders['tokens'].weight)
            print("vocab:", vocab)
            print("index to token vocab:", vocab.get_index_to_token_vocabulary(namespace='labels'))
            print("===MODEL DEBUG===")

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        
        if self.debug:
            for embedding in embeddings[0]: # Grab from first line in batch.
                print("First component:", embedding[0])

        class_avgs = self.class_avgs
        encoder_out = self.encoder(embeddings, labels, class_avgs, mask) # Modifies `class_avgs`.
        tag_logits = encoder_out
        # tag_logits = self.hidden2tag(encoder_out)
        if self.debug:
            print("===MODEL DEBUG===")
            # print(embeddings[0][0])
            print("Labels:", labels)
            print("Sentence:", sentence)
            print("Shape of embeddings:", embeddings.shape)
            print("encoder_out:", encoder_out)
            print("tag_logits:", tag_logits)
            print("tag_logits shape:", tag_logits.shape)
            print("===MODEL DEBUG===")
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
    logits = predictor.predict("I lived in *Munich last summer. \
                *Germany has a relaxing, slow summer lifestyle.")['tag_logits']
    # print("tag_logits:\n", np.array(tag_logits))
    tag_ids = np.argmax(logits, axis=-1)

    dataset_reader = InstDatasetReader()

    for instance in dataset_reader._read("../data/validate.txt"):
        tokenlist = list(instance['sentence'])
        labellist = list(instance['labels'])
        for i, token in enumerate(tokens):
            print(labellist[i], token)
    # for i, tag_id in enumerate(tag_ids):
    print("labels:", [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    shutil.rmtree(serialization_dir)
