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

# metric

## BoolQ

- Model: roberta-large
- GPU:
    - Tesla V100 16GB
    - Driver Version: 440.64.00
    - CUDA Version: 10.2
- Metric:
```json
{
  "best_epoch": 7,
  "peak_worker_0_memory_MB": 7449.71875,
  "peak_gpu_0_memory_MB": 9014.68115234375,
  "training_duration": "1:23:10.477994",
  "training_start_epoch": 0,
  "training_epochs": 9,
  "epoch": 9,
  "training_accuracy": 0.99459000742548,
  "training_loss": 0.018497146248644302,
  "training_worker_0_memory_MB": 7449.71875,
  "training_gpu_0_memory_MB": 9014.68115234375,
  "validation_accuracy": 0.8669724770642202,
  "validation_loss": 0.6529402052762915,
  "best_validation_accuracy": 0.8669724770642202,
  "best_validation_loss": 0.6136807546645776
}
```