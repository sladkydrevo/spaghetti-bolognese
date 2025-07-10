import os
import json

folder = r"C:\Users\RobosizeME\Documents\RAG-testing\dataset\questions"
eval_set = []

for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    with open(file_path, "r", encoding="utf8") as f:
        q = f.read()
        d = {
            "question": q,
            "answer": "",
            "source_file_name": file
        }
        eval_set.append(d)

with open(r"C:\Users\RobosizeME\Documents\spaghetti-bolognese\dataset\qa\qa_eval_set.json", "w", encoding="utf8") as f:
    json.dump(eval_set, f, indent=2, ensure_ascii=False)
