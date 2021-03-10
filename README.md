# SuperGLUE on AllenNLP

allennlp code for SuperGLUE benchmark.

# Usage

1. Step 1: create virtual environment & install dependencies

```shell
conda env create
python -m pip install -r requirments.txt
```

2. Step 2: train model
```shell
cd BoolQ
allennlp train -s exp/boolq training_config/boolq.jsonnet
```
