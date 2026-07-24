# CLI guide

The `tiny-lm` command-line interface supports four workflows:

- `train` creates a tokenizer and trains a new model.
- `resume` continues training from a checkpoint.
- `generate` produces text from a trained checkpoint.
- `evaluate` calculates loss and perplexity.

Run the top-level help command at any time:

```sh
tiny-lm --help
```

For command-specific options:

```sh
tiny-lm train --help
tiny-lm resume --help
tiny-lm generate --help
tiny-lm evaluate --help
```

## Setup

Create or activate the project virtual environment, then install the
dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

The CLI automatically selects the first available device in this order:

1. CUDA
2. Apple Metal (`mps`)
3. CPU

Use `--device` with any command to select a device explicitly:

```sh
tiny-lm generate checkpoints/latest.pt \
  --device cpu \
  --prompt "Once upon a time "
```

## Train

Start training with the defaults:

```sh
tiny-lm train
```

By default, the training data is read from `data/input.txt`, and the latest
state is written to `checkpoints/latest.pt`.

A shorter custom run:

```sh
tiny-lm train \
  --data-file /path/to/training_data.txt \
  --vocab-size 5000 \
  --seq-len 512 \
  --d-model 1024 \
  --num-heads 8 \
  --d-ff 2048 \
  --num-layers 2 \
  --batch-size 16 \
  --max-steps 1000 \
  --eval-interval 100 \
  --eval-batches 4 \
  --checkpoint checkpoints/experiment.pt \
  --checkpoint-interval 250
```

The input file must be UTF-8 text with one sample per line. The byte-level
tokenizer accepts every UTF-8 string, including unseen characters and emoji.

Useful training options:

| Option | Purpose |
| --- | --- |
| `--data-file PATH` | Select the training text file. |
| `--vocab-size N` | Set the target BPE vocabulary size (minimum 257). |
| `--seq-len N` | Set the maximum model context length. |
| `--batch-size N` | Set the number of sampled sequences per step. |
| `--max-steps N` | Set the total number of training steps. |
| `--eval-interval N` | Evaluate after every `N` steps. |
| `--seed N` | Reproduce model initialization and training batches. |
| `--checkpoint PATH` | Select the checkpoint output path. |
| `--checkpoint-interval N` | Save after every `N` steps. |
| `--compile` | Enable `torch.compile`. |
| `--no-compile` | Explicitly disable `torch.compile`. |

CLI options override corresponding `MINIBPE_*` environment variables.

## Resume

Continue an interrupted run using the target step count stored in its
checkpoint:

```sh
tiny-lm resume checkpoints/latest.pt
```

Extend a completed run to a higher total number of steps:

```sh
tiny-lm resume checkpoints/latest.pt --max-steps 250000
```

When the total step target changes, the CLI preserves optimizer momentum and
starts a fresh warmup/cosine learning-rate schedule for the additional steps.

To preserve the original checkpoint and write the resumed run elsewhere:

```sh
tiny-lm resume checkpoints/latest.pt \
  --max-steps 250000 \
  --output-checkpoint checkpoints/extended.pt
```

A checkpoint contains:

- model weights;
- tokenizer vocabulary;
- model and training configuration;
- optimizer, scheduler, and gradient-scaler state;
- completed step and recorded losses;
- random-number-generator state.

## Generate

Generate text from a checkpoint:

```sh
tiny-lm generate checkpoints/latest.pt \
  --prompt "Once upon a time " \
  --temperature 0.8 \
  --top-k 50 \
  --max-new-tokens 100
```

Generation options:

| Option | Purpose |
| --- | --- |
| `--prompt TEXT` | Required starting text. |
| `--temperature N` | Control randomness; lower values are more conservative. |
| `--top-k N` | Sample only from the `N` highest-scoring tokens. |
| `--max-new-tokens N` | Limit the number of generated tokens. |
| `--seed N` | Reproduce a sampling run. |

The prompt may contain any UTF-8 text.

## Evaluate

Evaluate the validation split saved in the checkpoint configuration:

```sh
tiny-lm evaluate checkpoints/latest.pt
```

The command prints average cross-entropy loss and perplexity:

```text
validation_loss=4.167429
validation_perplexity=64.549263
```

Evaluate a limited number of usable samples:

```sh
tiny-lm evaluate checkpoints/latest.pt \
  --split validation \
  --max-samples 100 \
  --batch-size 16
```

Evaluate the training split or a relocated dataset:

```sh
tiny-lm evaluate checkpoints/latest.pt \
  --split train \
  --data-file /new/path/to/training_data.txt
```

## Common errors

### `ModuleNotFoundError: No module named 'torch'`

Activate the virtual environment and install the requirements:

```sh
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

### Training data not found

Pass its location explicitly:

```sh
tiny-lm train --data-file /path/to/training_data.txt
```

### Resume target is not higher than the saved step

Inspect the error's checkpoint step, then provide a larger total:

```sh
tiny-lm resume checkpoints/latest.pt --max-steps 250000
```

### Out of memory

Lower the batch size, sequence length, or model dimensions:

```sh
tiny-lm train --batch-size 4 --seq-len 256 --d-model 512
```
