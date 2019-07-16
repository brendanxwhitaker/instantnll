import shutil
import tempfile

from allennlp.commands.train import train_model
from allennlp.common.params import Params

from model import InstEntityTagger # pylint: disable=unused-import
from predictor import InstPredictor
from dataset_reader import InstDatasetReader

def main():
    params = Params.from_file('../configs/elmo.jsonnet')
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

        reader = InstDatasetReader()
        predictor = InstPredictor(model, dataset_reader=InstDatasetReader())

        # Get test vocab.
        test_path = "../data/validate_cities.txt"
        # test_path = test_paths[i]
        test_dataset = reader.read(test_path) # Change to temp file.

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
                    print(tags[i] + str(token))

        # Allennlp seems to only support extending the vocabulary once.
        # This is a hack to modify the `old` number of embeddings at each iteration.
        extended_num_embeddings = len(model.vocab.get_index_to_token_vocabulary(namespace='tokens'))
        model.word_embeddings.token_embedder_tokens.num_embeddings = extended_num_embeddings


if __name__ == '__main__':
    main()
