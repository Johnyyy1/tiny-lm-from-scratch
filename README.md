# minibpetokenizer

Install the dependency in the project virtual environment:

```sh
python3 -m pip install -r requirements.txt
```

Then run:

```sh
python3 minibpe.py
```

The script automatically uses CUDA, Apple Metal (`mps`), or CPU in that order.
Configuration is available through environment variables:

```sh
MINIBPE_MAX_STEPS=1000 \
MINIBPE_BATCH_SIZE=16 \
MINIBPE_EVAL_INTERVAL=100 \
python3 minibpe.py
```

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
