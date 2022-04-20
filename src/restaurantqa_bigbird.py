# snippets of code are taken from https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Evaluating_Big_Bird_on_TriviaQA.ipynb

#!pip install datasets

from datasets import load_dataset

dataset = load_dataset("subjqa", "restaurants")

train = dataset['train']

dataset['train'][0]

"""Display random samples"""

from datasets import ClassLabel, Sequence
import random
import pandas as pd
from IPython.display import display, HTML

def show_random_elements(dataset, num_examples=3):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))

show_random_elements(train)

# This dataset only consists of data samples that do not exceed 4 * 4096 characters, which corresponds to BigBird's maximum length of 4096 tokens
# But this dataset's longest context length is only 921 tokens (for future use if the contexts exceed 4096 tokens)
dataset = dataset.filter(lambda x: (len(x['question']) + len(x['context'])) < 4 * 4096)

from transformers import BigBirdTokenizer, BigBirdForQuestionAnswering
import torch 

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-base-trivia-itc").to("cuda")

def get_best_valid_start_end_idx(start_scores, end_scores, example, top_k=1, max_size=100):
    best_start_scores, _ = torch.topk(start_scores, top_k)
    best_end_scores, best_end_idx = torch.topk(end_scores, top_k)

    best_start_idx = example['answers']['answer_start']
    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = torch.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = torch.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]

def evaluate(example):
    # encode question and context so that they are seperated by a tokenizer.sep_token and cut at max_length
    encoding = tokenizer(example["question"], example["context"], return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
    input_ids = encoding.input_ids.to("cuda")

    with torch.no_grad():
        start_scores, end_scores = model(input_ids=input_ids).to_tuple()
    start_score, end_score = get_best_valid_start_end_idx(start_scores[0], end_scores[0], example, top_k=8, max_size=16)
    #start_score, end_score = torch.argmax(start_scores), torch.argmin(end_scores)

    # Let's convert the input ids back to actual tokens 
    all_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    answer_tokens = all_tokens[start_score: end_score + 1]

    example["output"] = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    #.replace('"', '')  # remove space prepending space token and remove unnecessary '"'

    answers = example["targets"]
    predictions = example["output"]

    # if there is a common element, it's a match
    example["match"] = len(list(answers & predictions)) > 0

    return example

results = dataset.map(evaluate)
# print("Exact Match (EM): {:.2f}".format(100 * sum(results['match'])/len(results)))
