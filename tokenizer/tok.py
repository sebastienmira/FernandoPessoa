import json 
import regex as re

def get_pair_frequency(tokens, counts = None): #checks for the number of token pairs in the text. Returns dictionary with pair: # of occurrences
    counts={}  if counts is None else counts
    for pair in zip(tokens, tokens[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge_pair(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i< len(ids)-1 and pair[0] == ids[i] and pair[1] == ids[i+1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class Tokenizer():
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
    
    def train(self, vocab_size, text):
        tokens = text.encode('utf-8')
        for i in range(vocab_size - 256):
            pair_frequency = get_pair_frequency(tokens)
            top_pair = max(pair_frequency, key=pair_frequency.get) #returns pair that appear the most
            idx = len(self.vocab)
            print(f'Merging {top_pair} -> {idx}')
            self.merges[top_pair] = idx
            tokens = merge_pair(tokens, top_pair, idx)
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
    
    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        
        while len(tokens) > 1:
            pair_frequency = get_pair_frequency(tokens)
            pair = min(pair_frequency, key = lambda p:self.merges.get(p, float("inf"))) #returns pair that was merged first
            if pair not in self.merges:
                break
        
            idx = self.merges[pair] #get encoded id
            tokens = merge_pair(tokens, pair, idx)

        return tokens
    
    def decode(self, ids):
        tokens = b"".join([self.vocab[idx] for idx in ids]) #ids(list of integers as encoded) -> bytes
        text = tokens.decode("utf-8", errors='replace') #bytes -> characters
        return text


class RegexTokenizer(Tokenizer):
    def __init__(self):
        super().__init__() 
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  #GPT split pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {} # (int, int) -> int
        self.vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

    def train(self, text, vocab_size, verbose=False):
        
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)        
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        for i in range(num_merges):

            stats = {}

            for chunk_ids in ids:
                get_pair_frequency(chunk_ids, stats)

            top_pair = max(stats, key=stats.get) #returns most common pair
            idx = 256 + i
            
            if verbose:
                print(f'Merging {top_pair} into {idx}')
            
            ids = [merge_pair(chunk_ids, top_pair, idx) for chunk_ids in ids]
            
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids)>=2:
            stats = get_pair_frequency(ids)
            pair = min(stats, key=lambda p:self.merges.get(p, float("inf"))) #get the pair that has was merged first
            if pair not in self.merges:
                break #nothing else to merge
            
            idx = self.merges[pair]
            ids = merge_pair(ids, pair, idx)
        
        return ids

    def encode(self, text):
        # chunks encoded separately and then merged together
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        
        return ids

    def decode(self, ids):
        tokens = b"".join([self.vocab[idx] for idx in ids])
        text = tokens.decode("utf-8", errors='replace') #translates bytes to characters
        return text
