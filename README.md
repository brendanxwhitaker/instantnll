# instantnll

In development. 

## Abstract

The purpose of this project is to build a model which solves a highly simplified NER task from only one training example. It should be able to generalize within the constraints of the problem statement, and print relevant accuracy metrics upon evaluation. 

## Input

This model takes as input plaintext with some named entities prefixed with `*` and `!`. Thus, we limit the number of named entities we wish to classify to only 2. We are not doing any few/zero-shot learning. There are no unseen test classes. 

## TODO

Write some test cases for the `SimRel` module.
