
def main():

    reader = InstDatasetReader()
    train_dataset = reader.read(os.path.abspath('../data/train.txt'))
    validation_dataset = reader.read(os.path.abspath('../data/validate.txt'))
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    EMBEDDING_DIM = 6
    HIDDEN_DIM = 6

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    # Pretrained embedding. 
    pretrained_embedding = Embedding(pretrained_file=os.path.abspath('../../data/instantnll/GoogleNews-vectors-negative300.bin'))
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    model = EntityTagger(word_embeddings, lstm, vocab)
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=50,
                      cuda_device=cuda_device)
    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict("I lived in Munich last summer. Germany has a relaxing, slow summer lifestyle.")['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    # Here's how to save the model.
    with open("/tmp/model.th", 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files("/tmp/vocabulary")

    # And here's how to reload the model.
    vocab2 = Vocabulary.from_files("/tmp/vocabulary")
    model2 = LstmTagger(word_embeddings, lstm, vocab2)
    with open("/tmp/model.th", 'rb') as f:
        model2.load_state_dict(torch.load(f))
    if cuda_device > -1:
        model2.cuda(cuda_device)
    predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
    tag_logits2 = predictor2.predict("I lived in Munich last summer. Germany has a relaxing, slow summer lifestyle.")['tag_logits']
    np.testing.assert_array_almost_equal(tag_logits2, tag_logits)
