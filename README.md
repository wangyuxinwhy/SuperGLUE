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

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.64.00    Driver Version: 440.64.00    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:09.0 Off |                    0 |
| N/A   32C    P0    53W / 300W |      0MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

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