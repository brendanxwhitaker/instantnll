from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary

from instantnll.model import PosDatasetReader


import argparse

def main(args):
    
    # the token indexer is responsible for mapping tokens to integers
    token_indexer = ELMoTokenCharactersIndexer()

    reader = PosDatasetReader()

    train_dataset = reader.read(args.train_path)
    vocab = Vocabulary.from_instances(train_dataset)

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
     
    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})

if __name__ == 'main':
    parser = argparse.ArgumentParser(description='InstantLearn Solution.')
    parser.add_argument('train_path', type=str, help='Path to local training data.')
    args = parser.parse_args()
    main(args)
