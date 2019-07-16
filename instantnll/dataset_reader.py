from typing import Iterator, List, Dict

import logging
import torch

from overrides import overrides

from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
torch.manual_seed(1)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register('inst_dataset_reader')
class InstDatasetReader(DatasetReader):
    """
    DatasetReader for NER tagging data, one sentence per line, like
        The###DET dog###NN ate###V the###DET apple###NN

    Parameters
    ----------
    tokens_per_instance : ``int``, optional (default=``None``)
        If this is ``None``, we will have each training instance be a single sentence.  If this is
        not ``None``, we will instead take all sentences, including their start and stop tokens,
        line them up, and split the tokens into groups of this number, for more efficient training
        of the language model.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` representation will always be single token IDs - if you've specified
        a ``SingleIdTokenIndexer`` here, we use the first one you specify.  Otherwise, we create
        one with default parameters.
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for the text.  See :class:`Tokenizer`.
    """
    def __init__(self,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=False)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        print("Dataset Reader token indexers:", self._token_indexers)
        splitter = SimpleWordSplitter()
        self._tokenizer = tokenizer or WordTokenizer(word_splitter=splitter)
        self._tokens_per_instance = tokens_per_instance
        self._lower = False
        self.debug = False

    # pylint: disable=arguments-differ
    @overrides
    def text_to_instance(self, tokens: List[Token], ent_types: List[str] = None) -> Instance:
        if self._lower:
            tokens = [Token(str(token).lower()) for token in tokens]
        sentence_field = TextField(tokens, self._token_indexers)
        #===DEBUG===
        if self.debug:
            print("instantnll.dataset_reader.py:", sentence_field)
        fields = {"sentence": sentence_field}
        #===DEBUG===
        if ent_types:
            label_field = SequenceLabelField(labels=ent_types, sequence_field=sentence_field)
            fields["labels"] = label_field
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as text_file:
            instance_strings = text_file.readlines()
            if self._lower:
                instance_strings = [string.lower() for string in instance_strings]

        if self._tokens_per_instance is not None:
            all_text = " ".join([x.replace("\n", " ").strip() for x in instance_strings])
            tokenized_text = self._tokenizer.tokenize(all_text)
            num_tokens = self._tokens_per_instance + 1
            tokenized_strings = []
            logger.info("Creating dataset from all text in file: %s", file_path)
            for index in Tqdm.tqdm(range(0, len(tokenized_text) - num_tokens, num_tokens - 1)):
                tokenized_strings.append(tokenized_text[index:(index + num_tokens)])
        else:
            tokenized_strings = [self._tokenizer.tokenize(s) for s in instance_strings]

        for line in tokenized_strings:
            sentence = []
            ent_types = []
            for token in line: # Type: allennlp.data.tokenizers.token.Token
                token = str(token)
                ent_type = token[0]
                if ent_type not in ['!', '*']:
                    ent_type = '_'   # Indicates irrelevant non-tagged tokens.
                else:
                    token = token[1:]
                sentence.append(token)
                ent_types.append(ent_type)
            if sentence != []:
                yield self.text_to_instance([Token(word) for word in sentence], ent_types)
