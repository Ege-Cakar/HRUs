"""Fix ds by adding extra pipe"""

# <codecell>
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset, disable_caching
from transformers import AutoTokenizer
from tqdm import tqdm

disable_caching()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')

    ds_configs = [
        {'name': 'full', 'path': '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_full_text'},
        {'name': 'imply', 'path': '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_implies_text'},
        {'name': 'or', 'path': '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_or_par_text_32768'},
        {'name': 'php_par', 'path': '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_php_par_text_32768'},
        {'name': 'php_enum', 'path': '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_php_text_32768'},
    ]

    for conf in ds_configs:
        print(f"info: processing dataset {conf['name']}")
        dataset = DatasetDict.load_from_disk(conf['path'])
        dataset.clear_cache()

        # add extra sep
        dataset = dataset.map(lambda x: {'prompt': x['prompt'] + '|'}, num_proc=16)
        dataset = dataset.rename_column('completion', 'completion')
        
        dataset.save_to_disk(conf['path'] + '_pipe')
        del dataset


