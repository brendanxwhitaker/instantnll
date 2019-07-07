# instantnll

In development. 

## Abstract

The purpose of this project is to build a model which solves a highly simplified NER task from only one training example. It should be able to generalize within the constraints of the problem statement, and print relevant accuracy metrics upon evaluation. 

## Usage

The experiment configuration file can be found at `instantnll/experiment.jsonnet`. `<span style="color:blue"> span test</span>`

<pre>
{
    "dataset_reader":{
        "type": "instantnll",
    },
    "train_data_path":"<b>../data/train_small.txt</b>",
    "validation_data_path":"../data/validate_cities.txt",
    "model":{
        "type": "instantnll",
        "word_embeddings": {
            // Technically you could put a "type": "basic" here,
            // but that's the default TextFieldEmbedder, so doing so
            // is optional.
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "~/packages/data/instantnll/GoogleNews-vectors-negative300_SUBSET.txt",
                },
            }
        },
        "encoder": {
            "type": "CosineEncoder",
            "simrel": {
                "input_dim": 300,
                "num_classes": 3,
            },
        },
    },

    "iterator":{
        "type": "basic",
        "batch_size": 10,
    },
    "trainer":{
        "optimizer":{
            "type":"adam"
        },
        "num_epochs": 1,
    },
    "vocabulary":{
        "pretrained_files": {
            "tokens": "~/packages/data/instantnll/GoogleNews-vectors-negative300_SUBSET.txt",
        }
    },
}
</pre>


## Input

This model takes as input plaintext with some named entities prefixed with `*` and `!`. Thus, we limit the number of named entities we wish to classify to only 2. We are not doing any few/zero-shot learning. There are no unseen test classes. 

## TODO

Write some test cases for the `SimRel` module.

Write `Model` test cases. 

Add BERT support. 

Add character-level embedding support. 

Handle OOV words. 

Add `min_pretrained_embeddings` parameter for static embeddings. 
