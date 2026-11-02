import datasets
import random
def map_fn(example) :
    problem = example['Question'] 
    options = [example['Correct Answer'], example['Incorrect Answer 1'], example['Incorrect Answer 2'], example['Incorrect Answer 3']]
    #shuffle the options but find the correct answer
    random.shuffle(options)
    correct_answer = options.index(example['Correct Answer'])
    labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D'} 
    correct_answer = "Option " + str(labels[options.index(example['Correct Answer'])]) + f" which is [{example['Correct Answer']}]"

    problem_formatted = (f"{problem}\n"
                            f"A) {options[0]}\n"    
                            f"B) {options[1]}\n"
                            f"C) {options[2]}\n"
                            f"D) {options[3]}\n") 
    
    example['problem'] = problem_formatted
    example['answer'] = correct_answer
    # example['option'] = index
    if example["Writer's Difficulty Estimate"] is not None:
        if "undergraduate" in example["Writer's Difficulty Estimate"]: 
            example['difficulty'] = 1 
        elif "Post-graduate" in example["Writer's Difficulty Estimate"]:
            example['difficulty'] = 3 
        else:
            example['difficulty'] = 2 
    else:
        example['difficulty'] = 1
    return example

ds1 = datasets.load_dataset("Idavidrein/gpqa","gpqa_main",split='train')
print(ds1)

ds1 = ds1.map(map_fn)
ds1 = ds1.remove_columns([col for col in ds1.column_names if col not in ['problem', 'answer','option','difficulty','Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']])

#store dataset in hub as test set
ds = datasets.DatasetDict({'test': ds1})
## PUSH TO HF HUB IF REQUIRED
# ds.push_to_hub('mehuldamani/gpqa-eval', private=True) 
