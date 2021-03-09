# -*- coding: utf-8 -*-
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer

from boolq.dataset_reader import BoolQDatasetReader
from .const import TEST_FIXTURES_PATH


def test_boolq_dataset_reader_default_setting():
    reader = BoolQDatasetReader()
    data_path = str(TEST_FIXTURES_PATH / "toy_data.jsonl")
    print(data_path)
    instances = list(reader.read(data_path))

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["tokens"].tokens][:5] == [
        "Persian",
        "language",
        "--",
        "Persian",
        "(/ˈpɜːrʒən,",
    ]
    assert fields["label"].label == 1

    fields = instances[1].fields
    assert [t.text for t in fields["tokens"].tokens][:5] == [
        "Epsom",
        "railway",
        "station",
        "--",
        "Epsom",
    ]
    assert fields["label"].label == 0


def test_boolq_dataset_reader_roberta_setting():
    reader = BoolQDatasetReader(
        tokenizer=PretrainedTransformerTokenizer(
            "roberta-base", add_special_tokens=False
        ),
        token_indexers={"tokens": PretrainedTransformerIndexer("roberta-base")},
    )
    data_path = str(TEST_FIXTURES_PATH / "toy_data.jsonl")
    instances = list(reader.read(data_path))

    assert len(instances) == 2

    fields = instances[0].fields
    assert [t.text for t in fields["tokens"].tokens][:5] == [
        "<s>",
        "Pers",
        "ian",
        "Ġlanguage",
        "Ġ--",
    ]
    assert [t.text for t in fields["tokens"].tokens][-5:] == [
        "Ġspeak",
        "Ġthe",
        "Ġsame",
        "Ġlanguage",
        "</s>",
    ]
    assert fields["label"].label == 1

    fields = instances[1].fields
    assert [t.text for t in fields["tokens"].tokens][:5] == [
        "<s>",
        "E",
        "ps",
        "om",
        "Ġrailway",
    ]
    assert [t.text for t in fields["tokens"].tokens][-5:] == [
        "Ġe",
        "ps",
        "om",
        "Ġstation",
        "</s>",
    ]
    assert fields["label"].label == 0
