# instantnll

[![codecov](https://codecov.io/gh/langfield/instantnll/branch/dev/graph/badge.svg)](https://codecov.io/gh/langfield/instantnll) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langfield/instantnll/blob/master/instantnll/instant_colab.ipynb)

Frozen. 

## Abstract

The purpose of this project is to build a model which solves a highly simplified NER task from only one training example. It should be able to generalize within the constraints of the problem statement, and print relevant accuracy metrics upon evaluation. 

## Usage

The model can be run either from the command-line via the following command:

```
python3 run.py
```

while in the `<package_root>/instantnll/` directory, or by running the following notebook with jupyter:

```
instant.ipynb
```

### Configuration

When running from the command line, modify the appropriate fields in `experiment.jsonnet`. When running from `instant.ipynb`, modify the appropriate fields in `template.jsonnet`. The `<>_data_path` fields should be left blank when running from the notebook. The notebook will fill them automatically once the `Params` object has been created by pointing to temporary files created with the user input during execution of the notebook cells. 

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

## TODO

*In order of priority.*

Write `Model` test cases. 

Create notebook. DONE.  

Work on hosting embeddings (what's MTL?, how does mt-dnn do it?)

Write beamer slides. 

Add `min_pretrained_embeddings` parameter for static embeddings, i.e. add functionality so predictions can be run an arbitrary number of times on made-up-on-the-spot test data (currently, you must pass in the validation data with the train data so it can add it to the vocabulary, even though it never trains on it, since we don't add labels for the validation data). 

Add BERT support. 

Add character-level embedding support. 

Handle OOV words. 

