"""
Preprocess the glavieai reasoning dataset to parquet format
"""
import os
import datasets
import pandas as pd

from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs
from sklearn.model_selection import train_test_split
import argparse

def wrap_answer(text: str) -> str:
    # Find the position of </think>
    think_end = text.find('</think>')
    if think_end == -1:
        return text  # Return original text if </think> not found

    # Get the content after </think>
    content_start = think_end + len('</think>')
    content = text[content_start:].strip()

    # Construct the new string with answer tags
    result = text[:content_start] + '\n<answer>\n' + content + '\n</answer>'
    return result

def get_sequence_length(example, max_seq_len=4096):
    """
    Calculate the total length of prompt + response and filter if exceeding max_seq_len.
    """
    prompt = example['prompt']
    response = example['response']
    
    # Calculate total sequence length (you might want to use tokenizer-based length for more accuracy)
    total_length = len(prompt) + len(response)
    
    return total_length <= max_seq_len

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/glavieai")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--sample_size", default=300_000, type=int)
    parser.add_argument("--test_size", default=0.2, type=float, help="Proportion of dataset to include in test split")
    parser.add_argument("--max_seq_len", default=8192, type=int, help="Maximum sequence length of question + answer")

    args = parser.parse_args()

    data_source = "glaiveai/reasoning-v1-20m"
    dataset_raw = datasets.load_dataset(data_source, 'default', streaming=True)

    dataset_iterable = dataset_raw['train']
    sampled_iterable = dataset_iterable.shuffle(buffer_size=10_000, seed=42).take(args.sample_size)

    # Filter based on sequence length
    filtered_iterable = sampled_iterable.filter(
        lambda x: get_sequence_length(x, max_seq_len=args.max_seq_len)
    )

    sampled_list = list(filtered_iterable)
    sampled_dataset = Dataset.from_list(sampled_list)

    # Split the dataset into train and test sets
    train_list, test_list = train_test_split(sampled_list, test_size=args.test_size, random_state=42)
    train_dataset = Dataset.from_list(train_list)
    test_dataset = Dataset.from_list(test_list)
    print(f'train size: {train_dataset}')
    print(f'test size: {test_dataset}')

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('prompt')
            answer_raw = example.pop('response')
            solution = wrap_answer(answer_raw)

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question_raw}],
                "response": answer_raw,
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_processed = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_processed = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    os.makedirs(local_dir, exist_ok=True)
    train_processed.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_processed.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)