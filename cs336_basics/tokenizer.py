from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

try:
    from .pretokenization_example import find_chunk_boundaries
except ImportError:
    from pretokenization_example import find_chunk_boundaries
try:
    import regex as re
except ImportError:
    import re


# GPT-2 pretoken pattern
GPT2_PRETOKEN_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    return _BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


class _BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_token = vocab
        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}
        self.special_tokens = list(special_tokens or [])
        self.special_to_id = {tok: self.token_to_id[tok.encode("utf-8")] for tok in self.special_tokens}
        self._cache: dict[bytes, list[int]] = {}
        self._pretoken_re = re.compile(GPT2_PRETOKEN_PATTERN)
        self._special_re = self._compile_special_re(self.special_tokens)

    @staticmethod
    def _compile_special_re(special_tokens: list[str]):
        if special_tokens:
            escaped = [re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)]
            return re.compile("|".join(escaped))
        return None
    # re.escape() 函数用于转义字符串中的特殊字符，使其在正则表达式中被视为普通字符。
    def _encode_piece_bytes(self, piece_bytes: bytes) -> list[int]:
        cached = self._cache.get(piece_bytes)
        if cached is not None:
            return cached

        parts = [bytes([b]) for b in piece_bytes]
        while len(parts) > 1:
            best_rank = None
            best_pos = -1
            for i in range(len(parts) - 1):
                rank = self.merge_rank.get((parts[i], parts[i + 1]))
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pos = i
            if best_pos < 0:
                break
            parts = parts[:best_pos] + [parts[best_pos] + parts[best_pos + 1]] + parts[best_pos + 2 :]

        token_ids = [self.token_to_id[p] for p in parts]
        self._cache[piece_bytes] = token_ids
        return token_ids

    def _encode_ordinary_text(self, text: str) -> list[int]:
        out: list[int] = []
        for m in self._pretoken_re.finditer(text):
            out.extend(self._encode_piece_bytes(m.group(0).encode("utf-8")))
        return out

    def _encode_text(self, text: str) -> list[int]:
        if self._special_re is None:
            return self._encode_ordinary_text(text)

        out: list[int] = []
        cursor = 0
        for m in self._special_re.finditer(text):
            if m.start() > cursor:
                out.extend(self._encode_ordinary_text(text[cursor : m.start()]))
            out.append(self.special_to_id[m.group(0)])
            cursor = m.end()
        if cursor < len(text):
            out.extend(self._encode_ordinary_text(text[cursor:]))
        return out

    def encode(self, text: str) -> list[int]:
        return self._encode_text(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_token[i] for i in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:
            if chunk:
                yield from self._encode_text(chunk)


def _compile_pretoken_re(special_tokens: list[str]) -> re.Pattern:
    """Build one regex that matches special tokens first, then normal pretoken pieces."""
    ordered_special_tokens = sorted(special_tokens, key=len, reverse=True)
    if ordered_special_tokens:
        sp = "|".join(re.escape(t) for t in ordered_special_tokens)
        return re.compile(f"(?:{sp})|{GPT2_PRETOKEN_PATTERN}")
    return re.compile(GPT2_PRETOKEN_PATTERN)


def _collect_pretoken_counts(input_path: str, special_tokens: list[str]) -> Counter[bytes]:
    """Read corpus in chunks, pretokenize, accumulate piece counts in bytes."""
    # TODO: 复用你现在的 chunk + find_chunk_boundaries 框架
    with open (input_path, "rb") as f:
        pretoken = _compile_pretoken_re(special_tokens)
        boundaries = find_chunk_boundaries(f, 2048, b"<|endoftext|>")
        #Counter 本质上就是一个特殊的字典，专门用来做频率统计。
        counts = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            # TODO: chunk decode utf-8(ignore)
            chunk = (f.read(end - start)).decode("utf-8", errors="ignore")
            # TODO: 用 _compile_pretoken_re.finditer
            for m in pretoken.finditer(chunk):
                piece = m.group(0)
                # m.group(0) 是正则表达式匹配到的文本片段，
                # 即一个 pretoken piece。这个 piece 可能是一个普通的预分词单元，也可能是一个特殊 token。
                # TODO: 对普通 pretoken: counts[piece.encode('utf-8')] += 1
                if piece in special_tokens:
                    continue
                counts[piece.encode("utf-8")] += 1

    # TODO: 对 special token: 不进入训练词频（避免被 merge）
    return counts


def _init_vocab(special_tokens: list[str]) -> tuple[dict[int, bytes], dict[bytes, int], int]:
    """Initialize vocab with specials first, then 256 byte tokens."""
    vocab: dict[int, bytes] = {}
    token2id: dict[bytes, int] = {}
    next_id = 0

    # TODO: specials 先入 vocab（ID 从 0 开始）
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[next_id] = token_bytes
        token2id[token_bytes] = next_id
        next_id += 1
    # TODO: 再加入 bytes([0])..bytes([255])，跳过已存在 token
    for i in range(256):
        b = bytes([i])
        if b not in token2id:
            vocab[next_id] = b
            token2id[b] = next_id
            next_id += 1
        
    # TODO: 返回 vocab, token2id, next_id
    return vocab, token2id, next_id


def _build_word_freq(
    pretoken_counts: Counter[bytes], token2id: dict[bytes, int]
) -> dict[tuple[int, ...], int]:
    """Convert each pretoken bytes into tuple of byte-token ids, aggregate frequency."""
    # TODO: word = tuple(token2id[bytes([b])] for b in piece_bytes)
    from collections import defaultdict
    word_freq = defaultdict(int)


    for token_bytes, count in pretoken_counts.items():
        word = tuple(token2id[bytes([b])] for b in token_bytes)
        # TODO: word_freq[word] += cnt
        word_freq[word] += count
    return word_freq


def _count_pairs(
    word_freq: dict[tuple[int, ...], int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[tuple[int, ...]]]]:
    """Build pair frequency table and reverse index pair->words."""
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)

    # TODO: 遍历每个 word 的相邻 pair
    for word, freq in word_freq.items():
        for i in  range(len(word) - 1):
            pair = (word[i], word[i + 1])
    # TODO: pair_counts[pair] += word_count
            pair_counts[pair] += freq
    # TODO: pair_to_words[pair].add(word)
            pair_to_words[pair].add(word)
    return pair_counts, pair_to_words


def _choose_best_pair(
    pair_counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> tuple[int, int] | None:
    """Pick best pair with deterministic tie-break."""
    if not pair_counts:
        return None

    # TODO: 按 (频次最大, tie-break 固定) 选 pair
    best_pair = max(pair_counts.items(), key=lambda x: (x[1], vocab[x[0][0]] + vocab[x[0][1]]))
    return best_pair[0]
    # 建议 tie-break: (vocab[a] + vocab[b]) 的字节序，保证可复现


def _merge_one_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    """Replace all non-overlapping occurrences of pair in one word."""
    # TODO: 线性扫描 word，命中 pair -> 写入 new_id
    merge = []
    i = 0
    while i<len(word):
        if i<len(word)-1 and (word[i],word[i+1]) == pair:
            merge.append(new_id)
            i += 2
        else:
            merge.append(word[i])
            i += 1
    return tuple(merge)


def _apply_merge_incremental(
    pair: tuple[int, int],
    new_id: int,
    word_freq: dict[tuple[int, ...], int],
    pair_counts: dict[tuple[int, int], int],
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> None:
    """
    Incrementally update word_freq / pair_counts / pair_to_words after one merge.
    只更新受影响的 words，避免全量重算。
    """
    # TODO:
    # 1) affected_words = pair_to_words[pair]
    # 2) 对每个 old_word:
    #    - 从 pair_counts/pair_to_words 中减去 old_word 贡献
    #    - new_word = _merge_one_word(old_word, pair, new_id)
    #    - 写回 word_freq[new_word] += freq
    #    - 把 new_word 的 pair 贡献加回 pair_counts/pair_to_words
    # 3) 清理计数为 0 的 pair，避免表膨胀
    affected_words = pair_to_words[pair]
    for old_word in affected_words:
        freq = word_freq[old_word]
        # 从 pair_counts/pair_to_words 中减去 old_word 贡献
        for i in range(len(old_word)-1):
            p = (old_word[i], old_word[i+1])
            pair_counts[p] -= freq
            pair_to_words[p].remove(old_word)
        # new_word = _merge_one_word(old_word, pair, new_id)
        new_word = _merge_one_word(old_word, pair, new_id)
        # 写回 word_freq[new_word] += freq
        word_freq[new_word] += freq
        # 把 new_word 的 pair 贡献加回 pair_counts/pair_to_words
        for i in range(len(new_word)-1):
            p = (new_word[i], new_word[i+1])
            pair_counts[p] += freq
            pair_to_words[p].add(new_word)
    # 清理计数为 0 的 pair，避免表膨胀
    for p in list(pair_counts.keys()):
        if pair_counts[p] <= 0:
            del pair_counts[p]
            del pair_to_words[p]


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Return:
      vocab: id -> token_bytes
      merges: [(left_bytes, right_bytes), ...] in creation order
    """
    # 1) pretoken counts
    pretoken_counts = _collect_pretoken_counts(input_path, special_tokens)

    # 2) init vocab
    vocab, token2id, next_id = _init_vocab(special_tokens)

    # 3) build word freq
    word_freq = _build_word_freq(pretoken_counts, token2id)

    # 4) init pair stats
    pair_counts, pair_to_words = _count_pairs(word_freq)

    merges: list[tuple[bytes, bytes]] = []

    # 5) greedy merge loop
    while len(vocab) < vocab_size:
        best = _choose_best_pair(pair_counts, vocab)
        if best is None:
            break

        a, b = best
        new_token = vocab[a] + vocab[b]

        # 避免重复 token（通常不会发生，但建议防守）
        if new_token in token2id:
            # TODO: 处理重复 token 场景（可直接移除该 pair 后继续）
            del pair_counts[best]
            del pair_to_words[best]
            pass
        else:
            vocab[next_id] = new_token
            token2id[new_token] = next_id
            merges.append((vocab[a], vocab[b]))

            _apply_merge_incremental(
                best, next_id, word_freq, pair_counts, pair_to_words
            )
            next_id += 1

    return vocab, merges
