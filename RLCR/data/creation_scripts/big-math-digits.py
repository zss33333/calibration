import datasets
import random

ds1 = datasets.load_dataset("SynthLabsAI/Big-Math-RL-Verified",split='train')


def filter_func(example):
    ans = example['answer']
    solve_rate = example['llama8b_solve_rate'] 
    if solve_rate is None:
        return False
    if len(ans)<20 and 'text' not in ans and solve_rate > 0 and solve_rate < 0.75:
        #now I want to check if the answer is a rational number
        try:
            float(ans)
            return True
        except:
            return False
    else:
        return False

ds1 = ds1.filter(filter_func)

#subsample 2000 examples
ds1 = ds1.shuffle(seed=42)
ds1 = ds1.select(range(40000))
#first 30k examples are train, last 10k are test
ds = datasets.DatasetDict({'train': ds1.select(range(30000)), 'test': ds1.select(range(30000, 31000))})
## PUSH TO HF HUB IF REQUIRED
# ds.push_to_hub('mehuldamani/big-math-digits', private=True) 

