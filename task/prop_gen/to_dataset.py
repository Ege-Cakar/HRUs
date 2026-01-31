"""Format generated examples into HuggingFace datasets"""

# <codecell>
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset, disable_caching
from transformers import AutoTokenizer
from tqdm import tqdm

disable_caching()


def get_tokenizer():
    tok_path = Path(__file__).parent / 'data' / 'tok'
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def split_by_len(dataset):
    def make_entry(): return {True: [], False: []}

    splits = defaultdict(make_entry)
    print('info: splitting by length')
    for ex in tqdm(dataset):
        splits[ex['length']][ex['is_true']].append(ex)

    ds = {}
    print('info: assembling datasets')
    for key in tqdm(splits.keys()):
        for switch in [True, False]:
            bundle = splits[key][switch]
            if len(bundle) > 0:
                ds[f'{switch}_{key}'] = Dataset.from_list(splits[key][switch])

    return DatasetDict(ds)
    

def count_ops(dataset):
    def make_entry(): return {True: 0, False: 0}

    counts = defaultdict(make_entry)

    for ex in dataset:
        ops = frozenset(ex['ops'])
        counts[ops][ex['is_true']] += 1

    return counts

if __name__ == '__main__':
    maxlens= [32768]
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')

    # dataset = load_dataset('json', data_dir='data/raw_full', split='train', keep_in_memory=True, num_proc=16)
    # dataset = load_dataset('json', data_dir='data/raw_or', split='train', num_proc=16)
    # dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop/implies', split='train', num_proc=16)
    # dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop/php', split='train', num_proc=16)
    dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop/php_enum', split='train', num_proc=16)
    # dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop/or_par', split='train', num_proc=16)

    # add separator to distinguish prompt and completion
    dataset = dataset.map(lambda x: {'prompt': x['input'] + '|'}, remove_columns=['input'], num_proc=16)
    dataset = dataset.rename_column('proof', 'completion')
    # dataset = dataset.map(lambda x: {'completion': x['proof']}, num_proc=16)
    
    # maxlens = [1024]
    # dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop/or', split='train', keep_in_memory=True, num_proc=16)

    # if Path('data/tok').exists():
    #     tokenizer = AutoTokenizer.from_pretrained('data/tok')
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

    #     def make_corps(batch_size):
    #         for i in range(0, len(dataset), batch_size):
    #             inp = dataset[i:i+batch_size]['input']
    #             proof = dataset[i:i+batch_size]['proof']
    #             full = [a + b for a, b in zip(inp, proof)]
    #             yield full

    #     corps = make_corps(1000)

    #     print('info: training tokenizer...')
    #     tokenizer = tokenizer.train_new_from_iterator(corps, vocab_size=512)

    #     tokenizer.save_pretrained('data/tok')

    ds = split_by_len(dataset)

    for maxlen in maxlens:
        print(f'info: filtering by length {maxlen}')

        def filter_len(example):
            toks = tokenizer(example['prompt'] + example['completion'], return_attention_mask=False)
            return len(toks['input_ids']) <= maxlen
        
        filtered = ds.filter(filter_len, num_proc=16)
        ds_small = DatasetDict({name: subset for name, subset in filtered.items() if len(subset) > 0})
        ds_small.save_to_disk(f'/n/home09/wlt/scratch/data/prop_gen/data/hf_php_enum_text_{maxlen}')

    # ds.save_to_disk('/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_or_par_froz_text')

    # def to_toks(ex):
    #     inp_toks = tokenizer(ex['input'], return_attention_mask=False)
    #     inp_and_proof_toks = tokenizer(ex['input'] + ex['proof'], return_attention_mask=False)

    #     return {
    #         'input_ids': inp_toks['input_ids'],
    #         'full_ids': inp_and_proof_toks['input_ids'],
    #     }

    # ds = ds.map(to_toks, batched=False, num_proc=16)


