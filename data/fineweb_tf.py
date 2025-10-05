"""
FineWeb dataset loader in tensorflow
Downloads and tokenizes the data and saves data shards to disk.

"""
import os
import tensorflow as tf
import numpy as np
import tiktoken
import multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset

# initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    """Writes a numpy array of tokens to a binary file."""
    np.save(filename, tokens_np)

def main():
    local_dir = "edu_fineweb10B"
    remote_name = "sample-10BT"  
    shard_size = int(1e8)  # 100M tokens per shard

    # create a local directory
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train[:10%]")

    # tokenize all documents and write shards
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
        
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

    print("Done! Shards saved to", DATA_CACHE_DIR)
    
if __name__ == "__main__":
    mp.freeze_support()
    main()


def tf_dataset(shard_dir, batch_size, seq_len, split="train"):
    """Creates a TensorFlow dataset from the saved shards."""
    files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if split in f])
    
    datasets = []
    for fname in files:
        arr = np.load(fname, mmap_mode='r')
        x = tf.data.Dataset.from_tensor_slices(arr[:-1])
        y = tf.data.Dataset.from_tensor_slices(arr[1:])
        ds = tf.data.Dataset.zip((
            x.batch(seq_len, drop_remainder=True),
            y.batch(seq_len, drop_remainder=True)
        ))
        datasets.append(ds)
    
    ds = datasets[0]
    for extra in datasets[1:]:
        ds = ds.concatenate(extra)
        
    ds = ds.shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds

