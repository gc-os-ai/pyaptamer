# TODO: This module can be still refactored/improved.
__author__ = ["nennomp"]
__all__ = ["encode_protein", "rna2vec"]

from itertools import product

import torch
import numpy as np
from torch import Tensor


def encode_protein(
    device: torch.device, 
    target: str, 
    words: dict[str, int], 
    max_len: int,
) -> Tensor:
    """Encode a given protein.
    
    Args:
        target: Target protein.
        words: A dictionary containing (protein 3-mers, integer token IDs) pairs.

    Returns:
        A tensor containing the encoded target protein.
    """
    return torch.tensor(
        tokenize_sequences(
            seqset=[target], 
            words=words,
            max_len=max_len,
        ), 
        dtype=torch.int64).to(device)

def word2idx(word: str, words: dict[str, int]) -> int:
    if word in words.keys():
        return int(words[word])
    return 0

def dna2rna(seq: str) -> str:
    map = {'A':'A','C':'C','G':'G', 'U':'U', 'T':'U'}
    result = ''
    for s in seq:
        if s in map.keys():
            result += map[s]
        else:
            result += 'N'
            
    return result

def rna2vec(seqset):
    letters = ['A', 'C', 'G', 'U', 'N']

    words = {
        ''.join(triplet): i + 1 
        for i, triplet in enumerate(product(letters, repeat=3))
    }

    words = {word: i + 1 for i, word in enumerate(words)}

    outputs = []
    for seq in seqset:
        output = []

        converted_seq = dna2rna(seq)

        for i in range(0, len(converted_seq) - 2):  # -2 so we can index 3 letters
            output.append(word2idx(converted_seq[i:i + 3], words))

        if sum(output) != 0:
            # pad individual sequence
            padded_seq = np.pad(output, (0, 275 - len(output)), 'constant', constant_values=0)
            outputs.append(padded_seq)

    return np.array(outputs)

def tokenize_sequences(seqset, words, max_len: int, word_max_len: int = 3) -> np.ndarray:
    outputs = []
    for seq in seqset:
        output = []
        i = 0
        while i < len(seq): 
            flag=False
            for j in range(word_max_len, 0, -1): 
                if i+j <=len(seq): 
                    if word2idx(seq[i:i+j], words)!= 0: 
                        flag = True 
                        output.append(word2idx(seq[i:i+j], words)) 
                        if len(output)==max_len: 
                            outputs.append(np.array(output))
                            output = [] 
                        i+=j 
                        break 
            if flag==False:
                i+=1
                
        if len(output) != 0: 
            outputs.append(np.array(output))
        
    return pad_seq(outputs, max_len)

def pad_seq(dataset, max_len: int) -> np.ndarray:
    output = []
    for seq in dataset:
        pad = np.zeros(max_len)
        pad[:len(seq)] = seq
        output.append(pad)
        
    return np.array(output)