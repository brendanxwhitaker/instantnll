# instantnll

[![codecov](https://codecov.io/gh/brendanxwhitaker/instantnll/branch/dev/graph/badge.svg)](https://codecov.io/gh/brendanxwhitaker/instantnll)

In development. 

## Abstract

The purpose of this project is to build a model which solves a highly simplified NER task from only one training example. It should be able to generalize within the constraints of the problem statement, and print relevant accuracy metrics upon evaluation. 

## Usage

The experiment configuration file can be found at `<package_root>/instantnll/experiment.jsonnet`. A config file is shown below. The fields which need to be modified in order to run are shown in bold.  

<pre>
{
    "dataset_reader":{
        "type": "instantnll",
    },
    "train_data_path":"<b>../data/train_small.txt</b>",
    "validation_data_path":"<b>../data/validate_cities.txt</b>",
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
                    "embedding_dim": <b>300</b>,
                    "pretrained_file": "<b>~/packages/data/instantnll/GoogleNews-vectors-negative300_SUBSET.txt</b>",
                },
            }
        },
        "encoder": {
            "type": "CosineEncoder",
            "simrel": {
                "input_dim": <b>300</b>,
                "num_classes": <b>3</b>,
            },
        },
    },

    "iterator":{
        "type": "basic",
        "batch_size": <b>10</b>,
    },
    "trainer":{
        "optimizer":{
            "type":"adam"
        },
        "num_epochs": 1,
    },
    "vocabulary":{
        "pretrained_files": {
            "tokens": "<b>~/packages/data/instantnll/GoogleNews-vectors-negative300_SUBSET.txt</b>",
        }
    },
}
</pre>

### Input

This model takes as input plaintext with some named entities prefixed with `*` and `!`. Thus, we limit the number of named entities we wish to classify to only 2. We are not doing any few/zero-shot learning. There are no unseen test classes. 

*Note* The `num_classes` parameter must be set to the number of entity types `+1`, since we need a class for all other tokens.

The validation dataset should have tags as well if it is desired that the accuracy metric printed for the validation run be accurate. 

A small example dataset is given below:

```
I lived in *Munich last summer. *Germany has a relaxing, slow summer lifestyle. One night, I got food poisoning and couldn't find !Tylenol to make the pain go away, they insisted I take !aspirin instead.
``` 
Note the tags prepended to named entities of type `city` (`*`) and `drug` (`!`). 

### Running

Run with `python3 model.py` while in the `<package_root>/instantnll` directory. Currently all output is printed to console. No logging has been implemented yet, apart from that which is built into AllenNLP.  

## TODO

Write some test cases for the `SimRel` module.

Write `Model` test cases. 

Add BERT support. 

Add character-level embedding support. 

Handle OOV words. 

Add `min_pretrained_embeddings` parameter for static embeddings. 
