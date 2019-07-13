# Notes

The `batch_size` is measured in number of input sentences, which the `DatasetReader` determines by breaking on newlines. Empty lines count as sentences. Thus
```
I woke up this morning.

It was awful.
```
counts as three sentences, so with a `batch_size` of `3`, we would only need one batch for our entire input.

 
Perhaps it doesn't make much sense to train on the trivial class. Yes it doesn't make sense
to train on the trivial class, because then, in the case where the token currently being tagged
is farther from the trivial class avg vector than it is from the nontrivial class avg vectors,
it will label it as a nontrivial entity, even if it's really far away from everything. This is
bad behavior. So maybe a SimRel threshold parameter is a better approach.

The prediction block in `run.py` transforms the input such that we only ever have one batch. 

# Done 

Figure out exactly what the `hidden2tag` function should do, and if we need to compute logits from similarity values. Not needed. DONE.

Compute `class_avgs` by using a dict for the labels seen in vocab initialization in the `Model`. Perhaps pass an indicator dict to `CosineEncoder`, and then to `SimRel` which tells `SimRel` whether or not to just set the similarity to 1. This would treat the case where we haven't seen any examples of that class yet (similarity of a vector with nothing/itself should be 1). DONE. 

Question: is the above the best way to implement this? Maybe initialize the existing data structure to `float('-inf')` instead.  DONE. 

Figure out why allennlp's `cosine_similarity` returns a tensor of a weird dimension. I was passing multiple vectors. DONE. 

Add a tokenizer to `dataset_reader`. DONE.

Figure out what `tag_logits` should be.  DONE. 

Modify `model._hidden_to_tag()` so that it doesn't modify its argument. DONE.

Shift `encoder_out` so that all values are between 0 and 2 (currently -1 to 1). DONE. REVERTED.

Make `model` and `dataset_reader` cased.  DONE. 

Write a good battery of tests for the `SimRel` module. DONE. 

Change `class_avgs` from `List[torch.Tensor]` to `torch.Tensor`. DONE. REVERTED, bad idea.  

Normalize `tag_logits` by setting any negative similarity values to zero (worse than orthogonal is essentially useless to us). I.e. Apply relu. DONE.  

Write `EntityTagger.decode()` function. DONE.  

Put in notebook. DONE. 

Add color tag support in notebook output, and print nicely in a paragraph. DONE.  

Figure out how to extend vocabulary at test time.  DONE. 

# TODO

Write a prediction loop in run to continuously read in user input from command line, extend the vocabulary, and predict those new instances. 

Adapt the prediction loop for the notebook. 

Write test case for relu. 

Use F1 metric.

## Low priority  

Figure out what the intializer does in the `FeedForward` module test file. 

Figure out how `allennlp` is able to import absolute paths to their package submodules from within other submodules. 

Figure out why there is an extra namespace in the vocabulary called `tags` in addition to the one that I want (`labels`). 

Figure out when it is proper to use keyword arguments instead of positional arguments. Should I use them for every method I write?
    
Write SimRel test cases for negative values and higher sequence length, batch\_size. 

Write CosineEncoder test cases for mask. 

Initialize OOV words to the `0` vector instead of randomly. Then modify cosine distance to compute similarity of zero for zero vector, unless both vectors are zero, in which case it should be 1. 

Transform logits to `class_probabilities` via a softmax so that sim values form a probability distribution. 
