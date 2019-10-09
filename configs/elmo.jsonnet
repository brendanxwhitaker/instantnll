
{
    "dataset_reader":{
        "type": "inst_dataset_reader",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path":"../data/train_small.txt",
    "validation_data_path":"../data/validate_extended_tagged.txt",
    "model":{
        "type": "inst_entity_tagger",
        "word_embeddings": {
            // Technically you could put a "type": "basic" here,
            // but that's the default TextFieldEmbedder, so doing so
            // is optional.
            "type": "basic",
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.5
                },
            }
        },
        "encoder": {
            "type": "CosineEncoder",
            "simrel": {
                "input_dim": 1024,
                "num_classes": 3,
            },
        },
    },
    
    "iterator":{
        "type": "basic",
        "batch_size": 1,
    },
    "trainer":{
        "optimizer":{
            "type":"adam"
        },
        "num_epochs": 1,
    },
    "vocabulary":{
        "pretrained_files": {
            "elmo": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
        },
        "min_pretrained_embeddings": {
            "tokens": 1
        }
    },
}
