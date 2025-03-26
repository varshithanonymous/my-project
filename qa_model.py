import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import re

# Load pre-trained model and tokenizer
model_name = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Load passages from SQuAD TXT file
def load_squad_txt(file_path="squad_dataset.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    passages = re.split(r"\n\s*\n", text)  # Split paragraphs
    return passages

passage_db = load_squad_txt()

# Find the most relevant passage
def find_relevant_passage(question):
    for passage in passage_db:
        if any(word in passage.lower() for word in question.lower().split()):
            return passage
    return passage_db[0]  # Default to first passage if no match

# Answer extraction using BERT
def get_answer(question):
    passage = find_relevant_passage(question)
    
    inputs = tokenizer(question, passage, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))

    return answer if answer.strip() else "Sorry, I couldn't find an answer."

