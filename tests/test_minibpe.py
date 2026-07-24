from pathlib import Path

import pytest
import torch

from minibpe import (
    BPETokenizer,
    Config,
    EncodedDataset,
    MiniGPT,
    create_training_components,
    encode_dataset,
    estimate_loss,
    initialize_runtime,
    load_checkpoint,
    load_data,
    sample_batch,
    save_checkpoint,
)


def tiny_config(**overrides) -> Config:
    values = {
        "data_file": Path("data/input.txt"),
        "vocab_size": 257,
        "seq_len": 8,
        "d_model": 12,
        "num_heads": 3,
        "d_ff": 24,
        "num_layers": 2,
        "batch_size": 2,
        "max_steps": 4,
        "eval_interval": 2,
        "eval_batches": 2,
        "learning_rate": 3e-4,
        "dropout": 0.0,
        "max_token_length": 8,
        "compile_model": False,
        "seed": 42,
    }
    values.update(overrides)
    return Config(**values)


@pytest.fixture
def tokenizer() -> BPETokenizer:
    return BPETokenizer.train(["Hello world", "Hello there"], 270, 8)


def test_encode_decode_roundtrip(tokenizer: BPETokenizer) -> None:
    text = "Hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_text_roundtrip(tokenizer: BPETokenizer) -> None:
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_repeated_characters_roundtrip() -> None:
    tokenizer = BPETokenizer.train(["aaaaaaaa", "aaaaaa"], 260, 8)
    assert tokenizer.decode(tokenizer.encode("aaaaaa")) == "aaaaaa"


def test_encode_uses_learned_merge_order() -> None:
    tokenizer = BPETokenizer(
        {"^": 0, b"a": 1, b"b": 2, b"c": 3, b"bc": 4, b"ab": 5},
        [(2, 3, 4), (1, 2, 5)],
    )
    assert tokenizer.encode("abc") == [1, 4]


def test_encoding_is_deterministic(tokenizer: BPETokenizer) -> None:
    assert tokenizer.encode("Hello there") == tokenizer.encode("Hello there")


def test_tokenizer_serialization_preserves_merges() -> None:
    original = BPETokenizer(
        {"^": 0, b"a": 1, b"b": 2, b"c": 3, b"bc": 4, b"ab": 5},
        [(2, 3, 4), (1, 2, 5)],
    )
    restored = BPETokenizer.from_dict(original.to_dict())
    assert restored.merges == original.merges
    assert restored.encode("abc") == original.encode("abc")


def test_max_token_length_skips_invalid_pair() -> None:
    tokenizer = BPETokenizer.train(["aaaaaaaa", "bc"], 260, max_token_length=2)
    assert b"bc" in tokenizer.token_to_id
    assert all(len(token) <= 2 for token in tokenizer.token_to_id)


def test_byte_level_tokenizer_handles_unseen_unicode(tokenizer: BPETokenizer) -> None:
    text = "Příliš žluťoučký kůň 🐴"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_level_tokenizer_contains_every_byte(tokenizer: BPETokenizer) -> None:
    assert all(bytes((value,)) in tokenizer.token_to_id for value in range(256))


def test_model_logits_shape() -> None:
    model = MiniGPT(tiny_config(), vocab_size=8).eval()
    token_ids = torch.tensor([[1, 2, 3], [3, 2, 1]])
    logits = model(token_ids, torch.tensor([3, 3]))
    assert logits.shape == (2, 3, 8)


def test_model_cannot_see_future_tokens() -> None:
    model = MiniGPT(tiny_config(), vocab_size=8).eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = torch.tensor([[1, 2, 6, 7]])
    valid_lens = torch.tensor([4])
    first_logits = model(first, valid_lens)
    second_logits = model(second, valid_lens)
    torch.testing.assert_close(first_logits[:, :2], second_logits[:, :2])


def test_tied_embeddings_share_weights() -> None:
    model = MiniGPT(tiny_config(), vocab_size=8)
    assert model.output.weight is model.token_embedding.weight
    assert model.output.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_cached_and_uncached_logits_match() -> None:
    model = MiniGPT(tiny_config(), vocab_size=8).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(token_ids, torch.tensor([4]))

    caches = [None for _ in model.blocks]
    cached_logits = []
    for position in range(token_ids.shape[1]):
        token = token_ids[:, position : position + 1]
        position_id = torch.tensor([position])
        hidden = model.token_embedding(token) + model.position_embedding(position_id)
        new_caches = []
        for block, cache in zip(model.blocks, caches):
            hidden, cache = block.step(hidden, cache)
            new_caches.append(cache)
        caches = new_caches
        cached_logits.append(model.output(model.final_norm(hidden)))

    torch.testing.assert_close(torch.cat(cached_logits, dim=1), expected)


def test_same_seed_initializes_same_model() -> None:
    initialize_runtime(123, "cpu")
    first = MiniGPT(tiny_config(), vocab_size=8)
    initialize_runtime(123, "cpu")
    second = MiniGPT(tiny_config(), vocab_size=8)
    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)

    dataset = EncodedDataset(
        tokens=torch.tensor([[1, 2, 3], [3, 2, 1]]),
        valid_lens=torch.tensor([2, 2]),
    )
    first_batch = sample_batch(
        dataset,
        batch_size=4,
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
    )
    second_batch = sample_batch(
        dataset,
        batch_size=4,
        generator=torch.Generator().manual_seed(123),
        device=torch.device("cpu"),
    )
    for first_tensor, second_tensor in zip(first_batch, second_batch):
        torch.testing.assert_close(first_tensor, second_tensor)


def test_evaluation_does_not_advance_training_generator() -> None:
    config = tiny_config()
    model = MiniGPT(config, vocab_size=8)
    dataset = EncodedDataset(
        tokens=torch.tensor([[1, 2, 3], [3, 2, 1]]),
        valid_lens=torch.tensor([2, 2]),
    )
    train_generator = torch.Generator().manual_seed(config.seed)
    expected_state = train_generator.get_state().clone()
    eval_generator = torch.Generator().manual_seed(config.seed + 1)

    estimate_loss(model, dataset, config, eval_generator, torch.device("cpu"))

    assert torch.equal(train_generator.get_state(), expected_state)


def test_checkpoint_roundtrip_restores_training_state(tmp_path: Path) -> None:
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


def test_dataset_errors_are_clear(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_data(missing)

    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="is empty"):
        load_data(empty)


def test_dataset_packs_long_and_empty_lines_without_discarding() -> None:
    tokenizer = BPETokenizer.train(["abcdefgh", "", "ž"], 270, 8)
    dataset = encode_dataset(
        ["abcdefgh", "", "ž"],
        tokenizer,
        max_seq_len=4,
        name="test",
    )
    expected_stream = []
    for sample in ("abcdefgh", "", "ž"):
        expected_stream.extend(tokenizer.encode(sample, add_stop=True))

    reconstructed = []
    for index, (row, valid_len) in enumerate(zip(dataset.tokens, dataset.valid_lens)):
        window = row[: int(valid_len) + 1].tolist()
        reconstructed.extend(window if index == 0 else window[1:])

    assert reconstructed == expected_stream
    assert all(valid_len <= 4 for valid_len in dataset.valid_lens)


def test_adamw_hyperparameters_are_explicit() -> None:
    model = MiniGPT(tiny_config(), vocab_size=8)
    optimizer, _, _ = create_training_components(
        model,
        tiny_config(),
        torch.device("cpu"),
    )
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.95)
    assert group["weight_decay"] == 0.1


def test_invalid_head_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="must divide"):
        tiny_config(d_model=10, num_heads=3).validate()
