# TODO

Write a good battery of tests for the `SimRel` module. 

Write tests for `CosineEncoder`. 

Figure out what the intializer does in the `FeedForward` module test file. 

Figure out exactly what the `hidden2tag` function should do, and if we need to compute logits from similarity values.

Figure out how `allennlp` is able to import absolute paths to their package submodules from within other submodules. 

Figure out why there is an extra namespace in the vocabulary called `tags` in addition to the one that I want (`labels`). 
