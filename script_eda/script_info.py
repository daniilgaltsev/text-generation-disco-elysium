from typing import List, Optional, Tuple
import numpy as np
import transformers
import os


def get_script(script_path: Optional[str] = None) -> str:
    if script_path is None:
        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "text_generation", "data", "texts_extracted.txt")
    fd = open(script_path)
    extracted_script = fd.read()
    fd.close()
    return extracted_script


def get_lines(script: str) -> List[str]:
    lines = script.split('\n')
    return lines


def get_tokenizer() -> transformers.PreTrainedTokenizer:
    return transformers.GPT2Tokenizer.from_pretrained("gpt2")


def get_raw_text_stats(extracted_script: str, lines: List[str]) -> Tuple[int, int, int, int]:
    line_lengths = [len(line) for line in lines]
    max_length = max(line_lengths)
    words = extracted_script.split()

    max_length = len(lines[0])
    for i, line in enumerate(lines):
        length = len(line)
        if length > max_length:
            max_length = length
    return len(lines), len(words), len(extracted_script), max_length


def print_tokenized_stats(lines: List[str], tokenizer: transformers.PreTrainedTokenizer) -> None:
    encoded = tokenizer.batch_encode_plus(lines)
    input_ids = encoded['input_ids']
    lengths = np.array(sorted([[len(seq), idx] for idx, seq in enumerate(input_ids)]))
    percentiles = [50, 65, 80, 90, 95, 99]
    print("Token sequence length:\n\tmax = {},\twith {} sequences\t(idx={})".format(lengths[-1][0], len(lengths), lengths[-1][1]))
    for p in percentiles:
        print("\tat {}th percentile: {},\twith ~{} sequences left".format(p, int(np.percentile(lengths[:, 0], p)), int(len(lengths) * p / 100.0)))


if __name__ == "__main__":
    extracted_script = get_script()
    lines = get_lines(extracted_script)
    tokenizer = get_tokenizer()

    print("Number of lines: {},\nnumber of words: {},\nnumber of chars: {},\nlongest line: {}".format(
        *get_raw_text_stats(extracted_script, lines)
    ))
        
    print_tokenized_stats(lines, tokenizer)