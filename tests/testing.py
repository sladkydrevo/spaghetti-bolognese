import json
import tools.rag_functions as rf
import model_config.chroma_embedding_functions as chroma_embedding_functions


def load_json(path):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    return loaded_data
        
        
models_path = "models/models_test.json"
def list_models(models_path, models_type):
    models = load_json(models_path)
    return models[models_type]


embedding_models = list_models(models_path, models_type="embedding_models")
generative_models = list_models(models_path, models_type="generative_models")

    
def universal_ef(model_family, model_name):
    embedding_function = getattr(chroma_embedding_functions, model_family)
    ef = lambda: embedding_function(model_name)
    return ef

xd = universal_ef("sentence_transformers_embedding", "aaaaaaaaaaa")
print(xd)
"""
for model_family in embedding_models:
    for model_name in model_family:
        xd = universal_ef(model_family, model_name)"""