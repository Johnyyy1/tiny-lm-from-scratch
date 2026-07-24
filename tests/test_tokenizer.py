from tiny_lm.tokenizer import BPETokenizer


def test_encode_decode_roundtrip(tokenizer: BPETokenizer) -> None:
    text = "Hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_text_roundtrip(tokenizer: BPETokenizer) -> None:
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_stop_token_can_be_rendered_as_newline(tokenizer: BPETokenizer) -> None:
    token_ids = tokenizer.encode("first") + [tokenizer.stop_id] + tokenizer.encode("second")
    assert tokenizer.decode(token_ids, stop_replacement="\n") == "first\nsecond"


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
    restored = BPETokenizer.from_dict(original.to_dict(), original.merges)
    assert restored.merges == original.merges
    assert restored.encode("abc") == original.encode("abc")


def test_max_token_length_skips_invalid_pair() -> None:
    tokenizer = BPETokenizer.train(["aaaaaaaa", "bc"], 260, max_token_length=2)
    assert b"bc" in tokenizer.token_to_id
    assert all(isinstance(token, str) or len(token) <= 2 for token in tokenizer.token_to_id)


def test_byte_level_tokenizer_handles_unseen_unicode(tokenizer: BPETokenizer) -> None:
    text = "Příliš žluťoučký kůň 🐴"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_byte_level_tokenizer_contains_every_byte(tokenizer: BPETokenizer) -> None:
    assert all(bytes((value,)) in tokenizer.token_to_id for value in range(256))
