
from .pretokenization import (
    init_vocabulary,
    PreTokenizerArgs,
    pretokenize_file_parallel,
)
from .bpe import train_bpe_optimized, BPETokenizerParams


N_PROC = 8
TOKEN_SPLIT = "<|endoftext|>".encode("utf-8")
PRETOKEN_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
N_MAX_MERGES = 10_000


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    # 1. Vocabulary initialization
    vocabulary = init_vocabulary(special_tokens)
    # 2. Pre-tokenization
    pre_tok_params = PreTokenizerArgs(N_PROC, TOKEN_SPLIT, special_tokens, PRETOKEN_PAT)
    pre_tok_res = pretokenize_file_parallel(input_path, pre_tok_params)
    # 3. Compute BPE merges / Train BPE
    bpe_trained_res = train_bpe_optimized(vocabulary, pre_tok_res, N_MAX_MERGES)
    # bpe_trained_res = train_bpe_optimized_debug(vocabulary, pre_tok_res, N_MAX_MERGES)
    return bpe_trained_res
