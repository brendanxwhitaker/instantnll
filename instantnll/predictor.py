from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register('inst-predictor')
class InstPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SimpleWordSplitter()
        self._lower = False

    def predict(self, sentence: str) -> JsonDict:
        if self._lower:
            return self.predict_json({"sentence" : sentence.lower()})
        else:
            return self.predict_json({"sentence" : sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        print("instantnll.predictor: ", tokens)
        return self._dataset_reader.text_to_instance(tokens)
