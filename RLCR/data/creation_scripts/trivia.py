import datasets
import random

ds1 = datasets.load_dataset("mandarjoshi/trivia_qa","rc.nocontext",split='validation')

def map_func(example):
    ans = example['answer']
    aliases = ans['aliases'] 
    print(aliases)
    example['answer'] = ans
    return {"answer":aliases}

def filter_func(example):
    aliases = example['answer']['aliases']
    if len(aliases)<=5:
        return True
    else:
        return False

ds1 = ds1.filter(filter_func)
ds1 = ds1.map(map_func) 
#only keep question and answer columns
ds1 = ds1.remove_columns(['question_source','entity_pages','search_results'])

#subsample 2000 examples
ds1 = ds1.shuffle(seed=42)
ds1 = ds1.select(range(2000))
#store dataset in hub as test set
ds = datasets.DatasetDict({'test': ds1})

## PUSH TO HF HUB IF REQUIRED
# ds.push_to_hub('mehuldamani/trivia', private=True) 

