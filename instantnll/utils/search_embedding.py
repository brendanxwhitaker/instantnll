import sys
import pyemblib
from preprocessing import process_embedding

if __name__ == "__main__":
    vecs, labels = process_embedding(sys.argv[1], pyemblib.Format.Word2Vec, 1000000, None)
    with open('labels.txt', 'a') as the_file:
        for label in labels:
            the_file.write(label + '\n')
