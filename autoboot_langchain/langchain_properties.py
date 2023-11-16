
from autoboot.annotation.env import value_component


class LangchainProperties:
  
  @value_component("autoboot.langchain.llm.type")
  @staticmethod
  def llm_type():
    return "OpenAI"
  
  @value_component("autoboot.langchain.embeddings.type")
  @staticmethod
  def embeddings_type():
    return "OpenAI"
  
  @value_component("autoboot.langchain.vector_store.type")
  @staticmethod
  def vector_store_type():
    return "Chroma"
  
  @value_component("autoboot.langchain.vector_store.connection.host")
  @staticmethod
  def vector_store_connection_host():
    return "127.0.0.1"
  
  @value_component("autoboot.langchain.vector_store.connection.port")
  @staticmethod
  def vector_store_connection_port():
    return 19530
  
  @value_component("autoboot.langchain.vector_store.connection.url")
  @staticmethod
  def vector_store_connection_url():
    pass
  
  @value_component("autoboot.langchain.vector_store.connection.token")
  @staticmethod
  def vector_store_connection_token():
    pass
  
  @value_component("autoboot.langchain.vector_store.collection_name")
  @staticmethod
  def vector_store_collection_name():
    pass