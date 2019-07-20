import os
import sys
import shutil
import tempfile

from allennlp.commands.train import train_model
from allennlp.common.params import Params

from model import InstEntityTagger # pylint: disable=unused-import
from predictor import InstPredictor
from dataset_reader import InstDatasetReader

def main():
    params = Params.from_file('../configs/animals_money_classic_novalid.jsonnet')
    parms = params.duplicate()
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)
    shutil.rmtree(serialization_dir)
    print("Done training.")

    # Run predictions.
    if "validation_data_path" not in parms:
        # Grab pretrained file.
        vocab_params = parms.get(key="vocabulary")
        pretrained_files_params = vocab_params.get(key="pretrained_files")
        extension_pretrained_file = pretrained_files_params.get(key="tokens")
        
        # Construct the reader and predictor. 
        reader_parms = parms.pop('dataset_reader')
        reader_parms.pop('type')
        reader = InstDatasetReader.from_params(params=reader_parms)
        predictor = InstPredictor(model, dataset_reader=reader)

        # Get test vocab.
        test_path = "../data/animals_money_validate.txt"
        dataset_temp_dir = tempfile.mkdtemp()
        with open(test_path, "r") as text_file:
            lines = text_file.readlines()
        all_text = " ".join(lines) # Makes it all 1 batch.
        agg_path = os.path.join(dataset_temp_dir, "dataset_agg.txt")
        with open(agg_path, "w") as agg:
            agg.write(all_text)        
        test_dataset = reader.read(agg_path) # Change to temp file.

        # Find gold labels.
        print("instantnll.diff.py: test_dataset batch one labels:", test_dataset[0]['labels'])
        label_field = test_dataset[0]['labels'] # This is iterable.
        for label in label_field:
            print(label)

        # Extend vocabulary.
        embedding_sources_mapping = {"word_embeddings.token_embedder_tokens": extension_pretrained_file}
        model.vocab.extend_from_instances(params, test_dataset)
        model.extend_embedder_vocab(embedding_sources_mapping)

        print("Making preds.")
        # Make predictions
        with open(test_path, "r") as text_file:
            lines = text_file.readlines()
        all_text = " ".join(lines) # Makes it all 1 batch.
        output_dict = predictor.predict(all_text)
        tags = output_dict['tags']

        with open("log.log", 'a') as log:
            for instance in reader.read(test_path): # pylint: disable=protected-access
                tokenlist = list(instance['sentence'])
                for i, token in enumerate(tokenlist):
                    log.write(tags[i] + str(token) + "\n")
                    # print(tags[i] + str(token))

        # Allennlp seems to only support extending the vocabulary once.
        # This is a hack to modify the `old` number of embeddings at each iteration.
        extended_num_embeddings = len(model.vocab.get_index_to_token_vocabulary(namespace='tokens'))
        model.word_embeddings.token_embedder_tokens.num_embeddings = extended_num_embeddings
 
        shutil.rmtree(dataset_temp_dir)

if __name__ == '__main__':
    main()
