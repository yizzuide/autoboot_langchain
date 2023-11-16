
import os
from langchain.embeddings.base import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from autoboot.annotation import component
from autoboot_langchain.langchain_properties import LangchainProperties

@component(name="embeddings")
def get_embeddings() -> Embeddings:
  embeddings_type = LangchainProperties.embeddings_type()
  if(embeddings_type == "OpenAI"):
    return OpenAIEmbeddings()
  
@component(name="llm")
def get_llm() -> BaseLanguageModel:
  llm_type = LangchainProperties.llm_type()
  if(llm_type == "OpenAI"):
    return OpenAI(temperature=0)

@component(name="chat_llm")
def get_chat_llm() -> BaseLanguageModel:
  llm_type = LangchainProperties.llm_type()
  if(llm_type == "OpenAI"):
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
  