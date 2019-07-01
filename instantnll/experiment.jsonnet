{
    "dataset_reader":{
        "type": "instantnll",
    },
    "train_data_path":"../data/train.txt",
    "validation_data_path":"../data/validate.txt",
    "model":{
        "type": "instantnll",
        "word_embeddings": {
            // Technically you could put a "type": "basic" here,
            // but that's the default TextFieldEmbedder, so doing so
            // is optional.
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "~/packages/data/instantnll/GoogleNews-vectors-negative300.txt",
                },
            }
        },
        "encoder": {
            "type": "gru",
            "input_size": 210,
            "hidden_size": 300,
            "num_layers": 2,
            "dropout": 0.5,
            "bidirectional": true
        },
    },
    
    "iterator":{
        "type":"bucket"
    },
    "trainer":{
        "type":"default",
        "optimizer":{
            "type":"adam"
        }
    }
}
