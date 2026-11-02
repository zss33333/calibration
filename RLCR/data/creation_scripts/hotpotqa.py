import datasets 
import re
import string
import random

####THIS SCRIPT IS FOR HOTPOTQA MODIFIED ####
#### TO CREATE A DATASET FOR HOTPOTQA VANILLA, SET CHOICES = [0] ####
hotpotqa = datasets.load_dataset("hotpotqa/hotpot_qa","distractor")

def map_func(example):
    input_information = "" 
    sys = " Your answer will be verified with exact match score. To ensure correct verification, only provide the answer within the <answer> </answer> tags. Do not put any sentences or reasoning process within the <answer> </answer> tags."
    i =0 
    #randomly select number in  0,1,2,
    choices= [0,1,2]
    chosen_choice = random.choice(choices)
    supporting_titles= example['supporting_facts']["title"]
    chosen_title_to_remove = []
    try:
        if chosen_choice ==1:
            #sample one from supporting titles and one from non supporting titles, ensure they are different
            non_supporting_titles = list(set(example['context']['title']) - set(supporting_titles))
            chosen_title_to_remove = [random.choice(supporting_titles), random.choice(non_supporting_titles)]
        elif chosen_choice ==2:
            chosen_title_to_remove = supporting_titles
        else:
            #randomly select 2 titles to remove
            chosen_title_to_remove = random.sample(set(example['context']['title']) - set(supporting_titles), 2)
    except:
        print("Some error")
        chosen_choice, chosen_title_to_remove = 0, []

    for title, sentence_cluster in zip(example['context']['title'], example['context']['sentences']):
        if title not in chosen_title_to_remove:
            #sentence cluster is a list of sentences, convert it to a string
            sentence_cluster_str = "\n".join(sentence_cluster)
            input_information += f"Paragraph {i} \n\n {sentence_cluster_str} \n\n"
            i += 1
            input_information += "--------------------------------\n\n"
    

    if chosen_title_to_remove!= []:
        if i!=8:
            print('i is not 8')

    example['problem'] = f"Question: {example['question']} \n\n"
    example['problem'] += f"{sys} \n\n"
    example['problem'] += f"Supporting Information: {input_information} \n\n"
    example['source'] = 'hotpot'
    example['gold_removed'] = chosen_choice
    example['removed_titles'] = chosen_title_to_remove
    
    return example

def filter_func(example):
    return len(example['problem']) < 6000

hotpotqa = hotpotqa.map(map_func).filter(filter_func)
print(len(hotpotqa['train']))
print(len(hotpotqa['validation']))
#sample 10000 examples and filter length to be less than 10000 characters
train_set = hotpotqa['train'].select(range(20000))
test_set = hotpotqa['validation'].select(range(500))

#delete the question column
train_set = train_set.remove_columns('question')
test_set = test_set.remove_columns('question')

#Push to huggingface
ds_dict= datasets.DatasetDict({
    "train": train_set,
    "test": test_set
})

## PUSH TO HF HUB IF REQUIRED
# ds_dict.push_to_hub("mehuldamani/hotpot_qa", private=True)

