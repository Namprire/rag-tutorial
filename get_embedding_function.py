from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    #aws profile name and region name can be changed as per the user's aws configuration
    #change that to ollama
    #embeddings are important for the model to understand the context of the text
    #embeddings = BedrockEmbeddings(
    #    credentials_profile_name="default", region_name="us-east-1"
    #)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings