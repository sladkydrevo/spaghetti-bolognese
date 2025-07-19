import os
import csv
import json
from model_config.clients_api import clients
import model_config.chroma_embedding_functions as chroma_embedding_functions


def load_texts(folder_path):
    texts = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            text_data = {}
            file = os.path.join(folder_path, filename)
            filename = os.path.splitext(filename)[0]
            
            with open(file, "r", encoding="utf-8") as f:
                text_data["filename"] = filename
                text_data["text"] = f.read()
            texts.append(text_data)
            
    return texts


def chunk_texts(texts, chunk_size, overlap):
    text_chunks = {}
    for text_data in texts:
        filename = text_data["filename"]
        text = text_data["text"].split()
        chunk_id = 0
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_id += 1
            chunk = " ".join(text[i : i + chunk_size])
            chunk_name = f"{filename}_{chunk_id}"
            text_chunks[chunk_name] = chunk
            
    return text_chunks


def dict_to_kv_lists(data):
    names = list(data)
    texts = list(data.values())
    return names, texts


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    return loaded_data
        
        
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=5)
        
        
def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def list_models(models_path, models_type):
    models = load_json(models_path)
    return models[models_type]


def universal_ef(model_family, model_name):
    embedding_function = getattr(chroma_embedding_functions, model_family)
    ef = lambda: embedding_function(model_name)
    return ef


def vectorize_text(text, provider="openai"):
    client = clients[provider]()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding