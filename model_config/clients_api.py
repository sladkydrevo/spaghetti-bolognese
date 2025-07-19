import os

from together import Together
from openai import OpenAI

# keys have to match with models.json keys
clients = {
    "together" : lambda: Together(api_key=os.environ["TOGETHER_API_KEY"]),
    "openai" : lambda: OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
}