import shutil
import numpy as np
import tempfile

from allennlp.commands.train import train_model
from allennlp.common.params import Params

from model import InstEntityTagger
from predictor import InstPredictor
from dataset_reader import InstDatasetReader

def main():
    params = Params.from_file('experiment.jsonnet')
    parms = params.duplicate()
    serialization_dir = tempfile.mkdtemp()
    model = train_model(params, serialization_dir)
    
    predpath = parms.pop(key="validation_data_path")

    # Make predictions
    predictor = InstPredictor(model, dataset_reader=InstDatasetReader())
    with open(predpath, "r") as text_file:
        lines = text_file.readlines()
    all_text = " ".join(lines) # Makes it all 1 batch. 
    output_dict = predictor.predict(all_text)
    tags = output_dict['tags']
    dataset_reader = InstDatasetReader()
    
    with open("log.log", 'a') as log:
        for instance in dataset_reader._read(predpath):
            tokenlist = list(instance['sentence'])
            for i, token in enumerate(tokenlist):
                log.write(tags[i] + str(token) + "\n")
    shutil.rmtree(serialization_dir)

    
if __name__ == '__main__':
    main()
