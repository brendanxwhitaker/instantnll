from typing import Dict

import copy

import torch
import numpy as np


# from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits

from allennlp.training.metrics import CategoricalAccuracy

from encoder import CosineEncoder
from dataset_reader import InstDatasetReader
torch.manual_seed(1)

#========1=========2=========3=========4=========5=========6=========7=========8=========9=========0

@Model.register('inst_entity_tagger')
class InstEntityTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:

        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.vocab = vocab
        self.label_vocab = vocab.get_index_to_token_vocabulary(namespace='labels')

        inf_vec = torch.Tensor([float('-inf')] * encoder.get_input_dim())
        self.class_avgs = [inf_vec.clone() for i in range(len(self.label_vocab))]

        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))

        self.accuracy = CategoricalAccuracy()
        self.debug = True

        if self.debug:
            print("===MODEL DEBUG===")
            # print("Number of embeddings:", self.word_embeddings._token_embedders['tokens'].num_embeddings)
            # print("Embedding weights", self.word_embeddings._token_embedders['tokens'].weight)
            print("vocab:", vocab)
            print("index to token vocab:", vocab.get_index_to_token_vocabulary(namespace='labels'))
            print("===MODEL DEBUG===")

    #====1=========2=========3=========4=========5=========6=========7=========8=========9=========0

    def _hidden_to_tag(self, encoder_out: torch.Tensor) -> torch.Tensor:
        logits = [] 
        for i in range(len(encoder_out)):
            batch = []
            for j in range(len(encoder_out[0])):
                batch.append((encoder_out[i][j] + 1) / 2)
            logits.append(torch.stack(batch))
        return torch.stack(logits)

    #====1=========2=========3=========4=========5=========6=========7=========8=========9=========0

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)

        class_avgs = self.class_avgs
        encoder_out = self.encoder(embeddings, labels, class_avgs, mask) # Modifies `class_avgs`.
        
        tag_logits = encoder_out
        torch.nn.functional.relu(tag_logits, inplace=True)
        if self.debug:
            print("===MODEL DEBUG===")
            # print(embeddings[0][0])
            # print("Labels:", labels)
            # print("Sentence:", sentence)
            print("Shape of embeddings:", embeddings.shape)
            # print("encoder_out:", encoder_out)
            # print("tag_logits:", tag_logits)
            print("tag_logits shape:", tag_logits.shape)
            print("===MODEL DEBUG===")
        output_dict = {"tag_logits": tag_logits}
        # Should we do a softmax so that we get a probability distribution?

        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output_dict["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
        return output_dict
    
    #====1=========2=========3=========4=========5=========6=========7=========8=========9=========0

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        If there's a decode function, the predictor will try to use it.    
        """
        # Shape: (batch_size, seq_len, num_classes)
        logits = output_dict['tag_logits'] 
        argmax_indices = np.argmax(logits, axis=-1)
        all_tags = []
        for batch_argmaxs in argmax_indices: 
            label_vocab = self.vocab.get_index_to_token_vocabulary(namespace="labels")
            tags = [label_vocab[int(x)] for x in batch_argmaxs]
            all_tags.append(tags) 
        output_dict['tags'] = all_tags # Shape: (batch_size, seq_len)
        print("All_tags:", all_tags)
        return output_dict

    #====1=========2=========3=========4=========5=========6=========7=========8=========9=========0

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
