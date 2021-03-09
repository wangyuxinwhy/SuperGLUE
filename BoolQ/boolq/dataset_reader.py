# -*- coding: utf-8 -*-
import json
from typing import Optional, Iterable, Dict

from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Field
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField


@DatasetReader.register("boolq")
class BoolQDatasetReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_length: int = 115,
        **kwargs
    ):
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs
        )
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_length = max_length

    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in self.shard_iterable(f):
                record = json.loads(line.strip())
                yield self.text_to_instance(
                    passage=record.get("passage"),
                    question=record.get("question"),
                    label=record.get("label"),
                )

    def text_to_instance(  # type: ignore
        self, passage: str, question: str, label: Optional[bool] = None
    ) -> Instance:
        fields: Dict[str, Field] = {}
        passage_tokens = self.tokenizer.tokenize(passage)
        question_tokens = self.tokenizer.tokenize(question)
        tokens = self.tokenizer.add_special_tokens(passage_tokens, question_tokens)
        text_field = TextField(tokens)
        fields["tokens"] = text_field

        if label is not None:
            label_field = LabelField(int(label), skip_indexing=True)
            fields["label"] = label_field
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["tokens"].token_indexers = self.token_indexers  # type: ignore
