import os
import chromadb
import rag_functions as rf
from chromadb.utils import embedding_functions
import google.auth
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
)
from vertexai.language_models import TextEmbeddingModel
import vertexai


def sentence_transformers_embedding(model_name):
    st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    return st_ef


class VertexAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, project_id, location, model_name, dimensionality=3072):
        creds, _ = google.auth.default(quota_project_id=project_id)
        vertexai.init(project=project_id, location=location, credentials=creds)
        
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.dimensionality = dimensionality

    def __call__(self, texts):
        embeddings = []
        for text in texts:
            embedding = self.model.get_embeddings([text])
            embeddings.append(embedding[0].values)
        return embeddings
    
def vertex_embedding(model_name):
    vertex_ef = VertexAIEmbeddingFunction(
        project_id="nodal-vigil-455211-t6",
        location="us-central1",
        model_name=model_name,
    )
    return vertex_ef


def cohere_embedding(model_name):
    cohere_ef = embedding_functions.CohereEmbeddingFunction(
        api_key=os.environ["COHERE_API_KEY"], 
        model_name=model_name
    )
    return cohere_ef


def genai_embedding(model_name):
    genai_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.environ["GOOGLE_API_KEY"],
        model_name=model_name
    )
    return genai_ef


def jina_embedding(model_name):
    jina_ef = embedding_functions.JinaEmbeddingFunction(
        api_key=os.environ["JINA_API_KEY"],
        model_name=model_name,
    )
    return jina_ef


def openai_embedding(model_name):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name=model_name
    )
    return openai_ef