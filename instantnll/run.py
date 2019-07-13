import shutil
import tempfile

from allennlp.commands.train import train_model
from allennlp.common.params import Params
from allennlp.data import Vocabulary

from model import InstEntityTagger # pylint: disable=unused-import
from predictor import InstPredictor
from dataset_reader import InstDatasetReader

def main():
    params = Params.from_file('../configs/exper_novalid.jsonnet')
    parms = params.duplicate()
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)

    print("Done training:")
    test_path = "../data/validate.txt"
    train_path = parms.get(key="train_data_path")

    # Grab pretrained file.
    vocab_params = parms.get(key="vocabulary")
    pretrained_files_params = vocab_params.get(key="pretrained_files")
    extension_pretrained_file = pretrained_files_params.get(key="tokens")

    # Get test vocab.
    reader = InstDatasetReader()
    test_dataset = reader.read(test_path) # Change to temp file.
  
    embedding_sources_mapping = {"word_embeddings.token_embedder_tokens": extension_pretrained_file}
    model.vocab.extend_from_instances(params, test_dataset)
    model.extend_embedder_vocab(embedding_sources_mapping)
    
    print("Making preds.")
    # Make predictions
    predictor = InstPredictor(model, dataset_reader=InstDatasetReader())
    with open(test_path, "r") as text_file:
        lines = text_file.readlines()
    all_text = " ".join(lines) # Makes it all 1 batch.
    output_dict = predictor.predict(all_text)
    tags = output_dict['tags']
    dataset_reader = InstDatasetReader()

    with open("log.log", 'a') as log:
        for instance in dataset_reader._read(test_path): # pylint: disable=protected-access
            tokenlist = list(instance['sentence'])
            for i, token in enumerate(tokenlist):
                log.write(tags[i] + str(token) + "\n")
                print(tags[i] + str(token))
    shutil.rmtree(serialization_dir)

if __name__ == '__main__':
    main()
