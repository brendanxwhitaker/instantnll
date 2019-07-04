# TODO

Write a good battery of tests for the `SimRel` module. 

Write tests for `CosineEncoder`. 

Figure out what the intializer does in the `FeedForward` module test file. 

Figure out exactly what the `hidden2tag` function should do, and if we need to compute logits from similarity values.

Figure out how `allennlp` is able to import absolute paths to their package submodules from within other submodules. 

Figure out why there is an extra namespace in the vocabulary called `tags` in addition to the one that I want (`labels`). 

Compute `class_avgs` by using a dict for the labels seen in vocab initialization in the `Model`. Perhaps pass an indicator dict to `CosineEncoder`, and then to `SimRel` which tells `SimRel` whether or not to just set the similarity to 1. This would treat the case where we haven't seen any examples of that class yet (similarity of a vector with nothing/itself should be 1).

Question: is the above the best way to implement this? Maybe initialize the existing data structure to `float('-inf')` instead. 

Figure out why allennlp's `cosine_similarity` returns a tensor of a weird dimension. 
