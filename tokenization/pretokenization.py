# This file contains the pretokenization implementation
# copied from 1_tokenization.ipynb

import os
import regex as re
from dataclasses import dataclass
from collections import defaultdict
from multiprocessing import Pool

from aux.stanford_cs336.basics.pretokenization_example import find_chunk_boundaries


def init_vocabulary(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {x: bytes([x]) for x in range(256)}
    for spec_tok in special_tokens:
        vocab[len(vocab)] = spec_tok.encode("utf-8")
    return vocab


def remove_special_tokens(text: str, tokens: list[str]) -> str:
    # Create a regex pattern that matches all keys
    replacements = {tok: "" for tok in tokens}
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    # Use a lambda to replace each match with its corresponding value
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def count_tokens(text: str, pattern: str) -> dict[bytes, int]:
    token_count = {}
    for match in re.finditer(pattern, text):
        token = match.group()
        token_bytes = tuple(token.encode("utf-8"))
        token_count[token_bytes] = token_count.get(token_bytes, 0) + 1
    return token_count


def pretokenize(text: str, special_tokens: list[str], pretoken_pat: str) -> dict[tuple[bytes], int]:
    """Return bytes counts after special tokens removal and pre-tokenization."""
    text_clear = remove_special_tokens(text, special_tokens)
    token_count = count_tokens(text_clear, pretoken_pat)  # frequency table
    return token_count


@dataclass
class PreTokenizerArgs:
    n_proc: int
    token_split: str
    special_tokens: list[str]
    pretoken_pat: str


def pretokenize_file_parallel(filep: str, pt_args: PreTokenizerArgs) -> dict[tuple[bytes], int]:
    with open(filep, "rb") as file:
        bounds = find_chunk_boundaries(file, pt_args.n_proc, pt_args.token_split)
        # Create arguments for each chunk
        args = []
        for beg, end in zip(bounds[:-1], bounds[1:]):
            file.seek(beg)
            chunk = file.read(end - beg).decode("utf-8", errors="ignore")
            args.append((chunk, pt_args.special_tokens, pt_args.pretoken_pat))
    # Process chunks in parallel
    with Pool(processes=N_PROC) as pool:
        results = pool.starmap(pretokenize, args)
    # Reduce results
    pretoken_res = {}  # frequency table
    for chunk_res in results:
        for token_bytes, token_count in chunk_res.items():
            pretoken_res[token_bytes] = pretoken_res.get(token_bytes, 0) + token_count
    return pretoken_res
