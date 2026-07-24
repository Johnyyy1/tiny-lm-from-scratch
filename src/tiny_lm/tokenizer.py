"""UTF-8 byte-level byte-pair encoding."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Sequence

STOP_TOKEN = "^"
BYTE_VOCAB_SIZE = 256


class BPETokenizer:
    """A compact byte-level BPE tokenizer with complete UTF-8 coverage."""

    def __init__(
        self,
        token_to_id: dict[str | bytes, int],
        merges: Sequence[tuple[int, int, int]] | None = None,
    ):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        self.stop_id = token_to_id[STOP_TOKEN]
        self.byte_level = any(isinstance(token, bytes) for token in token_to_id)
        self.merges = list(merges) if merges is not None else self._infer_merges()
        self._merge_ranks = {
            (left_id, right_id): (rank, merged_id)
            for rank, (left_id, right_id, merged_id) in enumerate(self.merges)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def to_dict(self) -> dict[str | bytes, int]:
        """Serialize the token vocabulary."""

        return dict(self.token_to_id)

    @classmethod
    def from_dict(
        cls,
        values: dict[str | bytes, int],
        merges: Sequence[Sequence[int]] | None = None,
    ) -> BPETokenizer:
        """Restore a tokenizer from a serialized vocabulary and merge list."""

        token_to_id = {token: int(token_id) for token, token_id in values.items()}
        if token_to_id.get(STOP_TOKEN) != 0:
            raise ValueError("Checkpoint tokenizer has an invalid stop token.")
        if set(token_to_id.values()) != set(range(len(token_to_id))):
            raise ValueError("Checkpoint tokenizer IDs must be contiguous.")
        parsed_merges = None
        if merges is not None:
            parsed_merges = [tuple(int(value) for value in merge) for merge in merges]
            if any(len(merge) != 3 for merge in parsed_merges):
                raise ValueError("Checkpoint tokenizer has an invalid merge list.")
        return cls(token_to_id, parsed_merges)

    def _infer_merges(self) -> list[tuple[int, int, int]]:
        """Reconstruct merge rules from legacy vocab-only checkpoints."""

        merges: list[tuple[int, int, int]] = []
        ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for merged_id, merged_token in sorted(self.id_to_token.items()):
            if merged_token == STOP_TOKEN or len(merged_token) < 2:
                continue
            if isinstance(merged_token, bytes):
                base_tokens: Iterable[str | bytes] = (bytes((value,)) for value in merged_token)
            else:
                base_tokens = merged_token
            token_ids = [self.token_to_id[token] for token in base_tokens]
            token_ids = self._apply_merges(token_ids, ranks)
            if len(token_ids) != 2:
                raise ValueError(
                    f"Cannot reconstruct merge rule for vocabulary token {merged_token!r}."
                )
            left_id, right_id = token_ids
            merges.append((left_id, right_id, merged_id))
            ranks[(left_id, right_id)] = (len(merges) - 1, merged_id)
        return merges

    @staticmethod
    def _apply_merges(
        token_ids: list[int],
        merge_ranks: dict[tuple[int, int], tuple[int, int]],
    ) -> list[int]:
        while len(token_ids) > 1:
            pairs = (
                (token_ids[index], token_ids[index + 1]) for index in range(len(token_ids) - 1)
            )
            candidates = (
                (merge_ranks[pair][0], pair, merge_ranks[pair][1])
                for pair in pairs
                if pair in merge_ranks
            )
            try:
                _, best_pair, merged_id = min(candidates)
            except ValueError:
                break

            merged: list[int] = []
            index = 0
            while index < len(token_ids):
                if (
                    index + 1 < len(token_ids)
                    and (token_ids[index], token_ids[index + 1]) == best_pair
                ):
                    merged.append(merged_id)
                    index += 2
                else:
                    merged.append(token_ids[index])
                    index += 1
            token_ids = merged
        return token_ids

    @classmethod
    def train(
        cls,
        samples: Sequence[str],
        target_vocab_size: int,
        max_token_length: int,
    ) -> BPETokenizer:
        """Build a BPE vocabulary from text samples."""

        if not any(samples):
            raise ValueError("Cannot train a tokenizer on empty text.")
        token_to_id: dict[str | bytes, int] = {STOP_TOKEN: 0}
        token_to_id.update({bytes((value,)): value + 1 for value in range(BYTE_VOCAB_SIZE)})
        id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        merges: list[tuple[int, int, int]] = []
        recorded_pairs: set[tuple[int, int]] = set()

        if target_vocab_size < len(token_to_id):
            raise ValueError(
                "MINIBPE_VOCAB_SIZE must be at least the initial vocabulary "
                f"size ({len(token_to_id)})."
            )

        stop_id = token_to_id[STOP_TOKEN]
        tokens: list[int] = []
        for sample in samples:
            tokens.extend(token_to_id[bytes((value,))] for value in sample.encode("utf-8"))
            tokens.append(stop_id)

        token_count = len(tokens)
        previous = [index - 1 for index in range(token_count)]
        following = [index + 1 for index in range(token_count)]
        if token_count:
            previous[0] = -1
            following[-1] = -1
        alive = bytearray(b"\x01") * token_count

        pair_positions: dict[tuple[int, int], set[int]] = defaultdict(set)
        for left in range(token_count - 1):
            right = following[left]
            if tokens[left] != stop_id and tokens[right] != stop_id:
                pair_positions[(tokens[left], tokens[right])].add(left)

        versions: dict[tuple[int, int], int] = defaultdict(int)
        heap = [
            (-len(positions), 0, pair) for pair, positions in pair_positions.items() if positions
        ]
        heapq.heapify(heap)
        invalid_pairs: set[tuple[int, int]] = set()

        def current_pair(left: int) -> tuple[int, int] | None:
            if left < 0 or not alive[left]:
                return None
            right = following[left]
            if right < 0 or not alive[right]:
                return None
            if tokens[left] == stop_id or tokens[right] == stop_id:
                return None
            return tokens[left], tokens[right]

        def remove_pair(left: int, changed: set[tuple[int, int]]) -> None:
            pair = current_pair(left)
            if pair is not None:
                pair_positions[pair].discard(left)
                changed.add(pair)

        def add_pair(left: int, changed: set[tuple[int, int]]) -> None:
            pair = current_pair(left)
            if pair is not None:
                pair_positions[pair].add(left)
                changed.add(pair)

        while len(token_to_id) < target_vocab_size and heap:
            while heap:
                negative_count, version, pair = heapq.heappop(heap)
                if (
                    versions[pair] == version
                    and -negative_count == len(pair_positions[pair])
                    and pair_positions[pair]
                ):
                    break
            else:
                break

            left_token = id_to_token[pair[0]]
            right_token = id_to_token[pair[1]]
            if not isinstance(left_token, bytes) or not isinstance(right_token, bytes):
                raise TypeError("BPE attempted to merge a special token.")
            merged_token = left_token + right_token
            if len(merged_token) > max_token_length:
                invalid_pairs.add(pair)
                continue

            merged_id = token_to_id.get(merged_token)
            if merged_id is None:
                merged_id = len(token_to_id)
                token_to_id[merged_token] = merged_id
                id_to_token[merged_id] = merged_token
            if pair not in recorded_pairs:
                merges.append((pair[0], pair[1], merged_id))
                recorded_pairs.add(pair)

            changed_pairs: set[tuple[int, int]] = set()
            merged_occurrences = 0
            for left in sorted(pair_positions[pair]):
                if current_pair(left) != pair:
                    continue

                right = following[left]
                before = previous[left]
                after = following[right]
                remove_pair(before, changed_pairs)
                remove_pair(left, changed_pairs)
                remove_pair(right, changed_pairs)

                tokens[left] = merged_id
                following[left] = after
                if after >= 0:
                    previous[after] = left
                alive[right] = 0
                previous[right] = -1
                following[right] = -1

                add_pair(before, changed_pairs)
                add_pair(left, changed_pairs)
                merged_occurrences += 1

            for changed_pair in changed_pairs:
                versions[changed_pair] += 1
                count = len(pair_positions[changed_pair])
                if count and changed_pair not in invalid_pairs:
                    heapq.heappush(
                        heap,
                        (-count, versions[changed_pair], changed_pair),
                    )

            if merged_occurrences == 0:
                continue
        return cls(token_to_id, merges)

    def encode(self, text: str, add_stop: bool = False) -> list[int]:
        """Convert text into token IDs using learned merge order."""

        if self.byte_level:
            token_ids = [self.token_to_id[bytes((value,))] for value in text.encode("utf-8")]
        else:
            token_ids = []
            for position, char in enumerate(text):
                try:
                    token_ids.append(self.token_to_id[char])
                except KeyError:
                    raise ValueError(
                        f"Character {char!r} at position {position} "
                        "is not in the tokenizer vocabulary."
                    ) from None
        token_ids = self._apply_merges(token_ids, self._merge_ranks)
        if add_stop:
            token_ids.append(self.stop_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        stop_replacement: str = "",
    ) -> str:
        """Convert token IDs back into text.

        ``stop_replacement`` can render end-of-text boundaries as newlines when
        inspecting a generation that intentionally continues across them.
        """

        try:
            tokens = [self.id_to_token[token_id] for token_id in token_ids]
        except KeyError as exc:
            raise ValueError(f"Unknown token ID: {exc.args[0]}") from None
        if not self.byte_level:
            return "".join(
                stop_replacement if token == STOP_TOKEN else token
                for token in tokens
                if isinstance(token, str)
            )
        decoded = bytearray()
        for token in tokens:
            if isinstance(token, bytes):
                decoded.extend(token)
            elif token == STOP_TOKEN:
                decoded.extend(stop_replacement.encode("utf-8"))
        return decoded.decode("utf-8", errors="replace")
