import json

# Load the SQuAD dataset
with open("squad2.0.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

# Convert to text format
with open("squad_dataset.txt", "w", encoding="utf-8") as f:
    for data in squad_data["data"]:
        title = data["title"]
        f.write(f"Title: {title}\n")
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            f.write(f"Context: {context}\n")
            for qa in paragraph["qas"]:
                question = qa["question"]
                f.write(f"Question: {question}\n")
                for answer in qa["answers"]:
                    f.write(f"Answer: {answer['text']}\n")
                f.write("\n")

print("SQuAD dataset saved as text file.")
