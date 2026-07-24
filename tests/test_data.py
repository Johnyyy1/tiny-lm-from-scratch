from pathlib import Path

import pytest

from tiny_lm.data import encode_dataset, load_data
from tiny_lm.tokenizer import BPETokenizer


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
