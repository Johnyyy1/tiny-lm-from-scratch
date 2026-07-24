import torch

from tiny_lm.data import EncodedDataset, sample_batch
from tiny_lm.evaluation import estimate_loss
from tiny_lm.model import MiniGPT
from tiny_lm.training import create_training_components


def test_seeded_batch_sampling_is_deterministic() -> None:
    dataset = EncodedDataset(
        tokens=torch.tensor([[1, 2, 3], [3, 2, 1]]),
        valid_lens=torch.tensor([2, 2]),
    )
    first = sample_batch(
        dataset,
        batch_size=4,
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
    )
    second = sample_batch(
        dataset,
        batch_size=4,
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
    )
    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor, second_tensor)


def test_evaluation_does_not_advance_training_generator(tiny_config) -> None:
    config = tiny_config()
    model = MiniGPT(config, vocab_size=8)
    dataset = EncodedDataset(
        tokens=torch.tensor([[1, 2, 3], [3, 2, 1]]),
        valid_lens=torch.tensor([2, 2]),
    )
    train_generator = torch.Generator().manual_seed(config.seed)
    expected_state = train_generator.get_state().clone()

    estimate_loss(
        model,
        dataset,
        config,
        torch.Generator().manual_seed(config.seed + 1),
        torch.device("cpu"),
    )
    assert torch.equal(train_generator.get_state(), expected_state)


def test_adamw_hyperparameters_are_explicit(tiny_config) -> None:
    model = MiniGPT(tiny_config(), vocab_size=8)
    optimizer, _, _ = create_training_components(
        model,
        tiny_config(),
        torch.device("cpu"),
    )
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.95)
    assert group["weight_decay"] == 0.1
