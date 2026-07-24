# minibpetokenizer

See the complete [CLI guide](CLI.md) for setup, training, checkpoint resumption,
generation, evaluation, configuration, and troubleshooting.

Install the dependency in the project virtual environment:

```sh
python3 -m pip install -r requirements.txt
```

Train a new model:

```sh
python3 minibpe.py train
```

The script automatically uses CUDA, Apple Metal (`mps`), or CPU in that order.
CLI arguments override the corresponding environment variables:

```sh
python3 minibpe.py train \
  --max-steps 1000 \
  --batch-size 16 \
  --eval-interval 100 \
  --checkpoint checkpoints/latest.pt
```

Training writes a resumable checkpoint to `checkpoints/latest.pt` by default.
It contains the model, tokenizer, configuration, optimizer, scheduler, gradient
scaler, losses, current step, and random-number-generator states.

Resume an interrupted or completed run with a higher total step target:

```sh
python3 minibpe.py resume checkpoints/latest.pt --max-steps 200000
```

Generate text:

```sh
python3 minibpe.py generate checkpoints/latest.pt \
  --prompt "Once upon a time " \
  --temperature 0.8 \
  --top-k 50 \
  --max-new-tokens 100
```

Evaluate a checkpoint:

```sh
python3 minibpe.py evaluate checkpoints/latest.pt \
  --split validation \
  --batch-size 16
```

Run `python3 minibpe.py COMMAND --help` for all command-specific options.

Common settings and their defaults:

| Variable | Default |
| --- | ---: |
| `MINIBPE_DATA_FILE` | `/Users/jonas/Downloads/training_data.txt` |
| `MINIBPE_VOCAB_SIZE` | `5000` |
| `MINIBPE_SEQ_LEN` | `512` |
| `MINIBPE_D_MODEL` | `1024` |
| `MINIBPE_NUM_HEADS` | `8` |
| `MINIBPE_D_FF` | `2048` |
| `MINIBPE_NUM_LAYERS` | `2` |
| `MINIBPE_BATCH_SIZE` | `16` |
| `MINIBPE_MAX_STEPS` | `200000` |
| `MINIBPE_EVAL_INTERVAL` | `100` |
| `MINIBPE_EVAL_BATCHES` | `4` |
| `MINIBPE_COMPILE` | `0` |
