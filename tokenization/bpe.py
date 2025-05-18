# This file contains the BPE Training implementation
# copied from 1_tokenization.ipynb

from typing import Iterator
from dataclasses import dataclass
from collections import defaultdict
from tqdm.notebook import tqdm


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]            # index -> bytes
    merges: list[tuple[bytes, bytes]]  # index1,index2 -> new_index


def to_bytes(bytes_tuple: tuple[bytes]) -> Iterator[bytes]:
    return map(lambda x: bytes([x]) if isinstance(x, int) else bytes(x), bytes_tuple)


def merge_bytes(bytes_tuple: tuple[bytes], sep=b'') -> bytes:
    return sep.join(to_bytes(bytes_tuple))


def train_bpe_optimized(
    vocab_src: dict[int, bytes],
    freq_table_src: dict[tuple[bytes], int],
    num_merges: int,
    vocab_size: int = 10_000,
) -> BPETokenizerParams:
    """Optimized BPE training with incremental pair frequency updates.

    This is the implementation for debug purposes.
    
    Args:
        vocab_src: Initial vocabulary
        freq_table_src: Pre-tokenized vocabulary with frequencies: {(l,o,w): 5, ...}
        num_merges: Number of merge operations to perform
        
    Returns:
        BPETokenizerParams consisting of learned vocab and list of merge operations
            in order they were learned
    """
    vocab = vocab_src.copy()
    freq_table = freq_table_src.copy()
    # Init data structures
    merges = []
    pair_counts = defaultdict(int)
    pair_locations = defaultdict(set)
    # Precompute all initial pairs and their locations
    for token, freq in freq_table.items():
        for pair in zip(token, token[1:]):
            pair_counts[pair] += freq
            pair_locations[pair].add(token)
    # Train tokenizer
    for merge_idx in range(num_merges):
        if len(pair_counts) == 0 or len(vocab) >= vocab_size:
            # stop if all possible pairs were merged or met vocab_size threshold
            msgs = ['All possible pairs were merged', 'Met vocab_size threshold']
            msg = msgs[0] if len(pair_counts) == 0 else msgs[1]
            print(msg)
            break
        # TODO: Use heap instead of max
        pair_to_merge = max(pair_counts.items(), key=lambda p_cnt: (p_cnt[1], merge_bytes(p_cnt[0], b"|")))[0]
        # Update structures
        merges.append(tuple(to_bytes(pair_to_merge)))
        merged = merge_bytes(pair_to_merge)
        vocab[len(vocab)] = merged
        first, second = pair_to_merge
        # Update frequency table and tracking structures
        affected_tokens = pair_locations[pair_to_merge].copy()
        pairs_to_remove = set()  # Non existing pairs after merge
        for old_token in affected_tokens:
            # Merge and obtain new_token
            new_token = []
            pi = 0
            while pi < len(old_token):
                if pi < len(old_token)-1 and old_token[pi] == first and old_token[pi+1] == second:
                    new_token.append(merged)
                    pi += 2
                else:
                    new_token.append(old_token[pi])
                    pi += 1
            new_token = tuple(new_token)
            # Initialize frequency table for new token
            freq_table[new_token] = freq_table[old_token]
            # Remove old token from frequency table
            del freq_table[old_token]
            # Update pairs_count and pair_locations for new token
            old_pairs = set()  # Old pairs that don't intersect with merged
            for pair in zip(new_token, new_token[1:]):
                if pair[0] == merged and pair[1] == merged:
                    # merged pairs are joint: (re, re)
                    # rem_pair
                    rem_pair = (second, first)
                    pair_counts[rem_pair] -= freq_table[new_token]
                    if pair_counts[rem_pair] == 0:
                        pairs_to_remove.add(rem_pair)
                    # new_pair
                    new_pair = pair
                    pair_counts[new_pair] += freq_table[new_token]
                    pair_locations[new_pair].add(new_token)
                elif pair[0] == merged or pair[1] == merged:
                    # pairs that intersect with merged: (re, a)
                    # rem_pair
                    rem_pair = (second, pair[1]) if pair[0] == merged else (pair[0], first)
                    pair_counts[rem_pair] -= freq_table[new_token]
                    if pair_counts[rem_pair] == 0:
                        pairs_to_remove.add(rem_pair)
                    # new_pair
                    new_pair = (merged, pair[1]) if pair[0] == merged else (pair[0], merged)
                    pair_counts[new_pair] += freq_table[new_token]
                    pair_locations[new_pair].add(new_token)
                else:
                    # old pairs that don't intersect with merged: (a, b)
                    old_pairs.add(pair)
            # Update old pairs locations
            for old_pair in old_pairs:
                pair_locations[old_pair].remove(old_token)
                pair_locations[old_pair].add(new_token)
        # Remove non existing pairs
        for rem_pair in pairs_to_remove:
            del pair_counts[rem_pair]
            del pair_locations[rem_pair]
        del pair_counts[pair_to_merge]
        del pair_locations[pair_to_merge]
    res = BPETokenizerParams(vocab, merges)
    return res


# DEBUG IMPLEMENTATION


def print_debug_msg(msg: str):
    print()
    print('='*60)
    print(msg)
    print()


def print_debug_structs(
    freq_table: dict[tuple[bytes], int],
    pair_counts: dict[tuple[bytes], int],
    pair_locations: dict[tuple[bytes], set[tuple[bytes]]],
):
    # print frequency table
    print('freq_table:')
    _t_freq_table_cnt = {tuple(to_bytes(pair)): cnt for pair, cnt in freq_table.items()}
    print(_t_freq_table_cnt)
    print()
    # print pair count
    print('pair_counts:')
    _t_pair_cnt = {merge_bytes(pair, b"|"): cnt for pair, cnt in pair_counts.items()}
    print(_t_pair_cnt)
    print()
    # print pair locations
    print('pair_locations:')
    _t_pair_locations_f = {
        merge_bytes(pair, b"|"): [tuple(to_bytes(loc)) for loc in locs]
        for pair, locs in pair_locations.items()
    }
    print(f'{_t_pair_locations_f}')
    print()


def train_bpe_optimized_debug(
    vocab_src: dict[int, bytes],
    freq_table_src: dict[tuple[bytes], int],
    num_merges: int,
    vocab_size: int = 10_000,
) -> BPETokenizerParams:
    """Optimized BPE training with incremental pair frequency updates.

    This is the implementation for debug purposes.
    
    Args:
        vocab_src: Initial vocabulary
        freq_table_src: Pre-tokenized vocabulary with frequencies: {(l,o,w): 5, ...}
        num_merges: Number of merge operations to perform
        
    Returns:
        BPETokenizerParams consisting of learned vocab and list of merge operations
            in order they were learned
    """
    vocab = vocab_src.copy()
    freq_table = freq_table_src.copy()
    # Init data structures
    merges = []
    pair_counts = defaultdict(int)
    pair_locations = defaultdict(set)
    # Precompute all initial pairs and their locations
    for token, freq in freq_table.items():
        for pair in zip(token, token[1:]):
            pair_counts[pair] += freq
            pair_locations[pair].add(token)

    for merge_idx in range(num_merges):
        
        if len(pair_counts) == 0:
            # stop if all possible pairs were merged
            print_debug_msg('>>> All possible pairs were merged')
            print_debug_structs(freq_table, pair_counts, pair_locations)
            break

        if len(vocab) >= vocab_size:
            # stop if met vocab_size threshold
            print_debug_msg('>>> Met vocab_size threshold')
            print_debug_structs(freq_table, pair_counts, pair_locations)
            break
        
        print_debug_msg(f'>>> Merge ({merge_idx+1})')
        print_debug_structs(freq_table, pair_counts, pair_locations)
        
        # TODO: Use heap instead of max
        # pair_to_merge = max(pair_counts.items(), key=lambda p_cnt: (p_cnt[1], p_cnt[0]))[0]
        pair_to_merge = max(pair_counts.items(), key=lambda p_cnt: (p_cnt[1], merge_bytes(p_cnt[0], b"|")))[0]

        print(f'pair_to_merge: {merge_bytes(pair_to_merge, b"|")}')
    
        # Update structures
        merges.append(tuple(to_bytes(pair_to_merge)))
        merged = merge_bytes(pair_to_merge)
        vocab[len(vocab)] = merged
        first, second = pair_to_merge
        print(f'merged_pair: {merged}')
    
        print()
        
        # Update frequency table and tracking structures
        affected_tokens = pair_locations[pair_to_merge].copy()
        pairs_to_remove = set()  # Non existing pairs after merge
        for old_token in affected_tokens:
            print(f'old_token: {tuple(to_bytes(old_token))}')
            # Merge and obtain new_token
            new_token = []
            pi = 0
            while pi < len(old_token):
                if pi < len(old_token)-1 and old_token[pi] == first and old_token[pi+1] == second:
                    new_token.append(merged)
                    pi += 2
                else:
                    new_token.append(old_token[pi])
                    pi += 1

            new_token = tuple(new_token)
            print(f'  new_token: {tuple(to_bytes(new_token))}')
            
            # Initialize frequency table for new token
            freq_table[new_token] = freq_table[old_token]
            # Remove old token from frequency table
            del freq_table[old_token]

            # Update pairs_count and pair_locations for new token
            old_pairs = set()  # Old pairs that don't intersect with merged
            for pair in zip(new_token, new_token[1:]):
                if pair[0] == merged and pair[1] == merged:
                    # merged pairs are joint: (re, re)
                    # rem_pair
                    rem_pair = (second, first)
                    pair_counts[rem_pair] -= freq_table[new_token]
                    if pair_counts[rem_pair] == 0:
                        pairs_to_remove.add(rem_pair)
                    # new_pair
                    new_pair = pair
                    pair_counts[new_pair] += freq_table[new_token]
                    pair_locations[new_pair].add(new_token)
                elif pair[0] == merged or pair[1] == merged:
                    # pairs that intersect with merged: (re, a)
                    # rem_pair
                    rem_pair = (second, pair[1]) if pair[0] == merged else (pair[0], first)
                    pair_counts[rem_pair] -= freq_table[new_token]
                    if pair_counts[rem_pair] == 0:
                        pairs_to_remove.add(rem_pair)
                    # new_pair
                    new_pair = (merged, pair[1]) if pair[0] == merged else (pair[0], merged)
                    pair_counts[new_pair] += freq_table[new_token]
                    pair_locations[new_pair].add(new_token)
                else:
                    # old pairs that don't intersect with merged: (a, b)
                    old_pairs.add(pair)
            # Update old pairs locations
            for old_pair in old_pairs:
                pair_locations[old_pair].remove(old_token)
                pair_locations[old_pair].add(new_token)

        # Remove non existing pairs
        for rem_pair in pairs_to_remove:
            print(f'    rem_pair: {merge_bytes(rem_pair, b"|")}')
            del pair_counts[rem_pair]
            del pair_locations[rem_pair]
        del pair_counts[pair_to_merge]
        del pair_locations[pair_to_merge]
    
    res = BPETokenizerParams(vocab, merges)
    return res
