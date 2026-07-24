from pathlib import Path

import torch

from tiny_lm.checkpoint import load_checkpoint, save_checkpoint
from tiny_lm.model import MiniGPT
from tiny_lm.tokenizer import BPETokenizer
from tiny_lm.training import create_training_components


def test_checkpoint_roundtrip_restores_training_state(
    tmp_path: Path,
    tiny_config,
) -> None:
    config = tiny_config()
    tokenizer = BPETokenizer.train(["abcd", "bcda"], 264, 8)
    model = MiniGPT(config, tokenizer.vocab_size)
    optimizer, scheduler, scaler = create_training_components(
        model,
        config,
        torch.device("cpu"),
    )
    loss = model(torch.tensor([[1, 2]]), torch.tensor([2])).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        config,
        optimizer,
        scheduler,
        scaler,
        step=1,
        training_losses=[1.5],
        validation_losses=[1.75],
        train_generator=torch.Generator().manual_seed(config.seed),
        eval_generator=torch.Generator().manual_seed(config.seed + 1),
    )
    checkpoint = load_checkpoint(checkpoint_path)

    assert checkpoint["step"] == 1
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["training_losses"] == [1.5]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(checkpoint["model_state_dict"][name], value)
