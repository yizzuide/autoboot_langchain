
from typing import List, Type
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores import VectorStore, Milvus
from langchain.retrievers import MilvusRetriever
from autoboot.annotation import component
from autoboot_langchain.langchain_properties import LangchainProperties
from .vector_ops import VectorOps

class MilvusOps(VectorOps):
  
  def get_type(self) -> Type[VectorStore]:
    return Milvus
  
  def get_connection_args(self) -> dict:
    return {
      "uri": LangchainProperties.vector_store_connection_url(),
      "token": LangchainProperties.vector_store_connection_token(),
      "secure": True
    }

  def do_store_index(self, docs: List[Document], embedding: Embeddings) -> VectorStore:
    return Milvus.from_documents(
      docs,
      embedding,
      connection_args=self.get_connection_args(),
      collection_name=LangchainProperties.vector_store_collection_name()
    )

  @component(name="milvus_store_retriever")
  def create_store_retriever(self) -> VectorStoreRetriever:
    return MilvusRetriever(connection_args=self.get_connection_args(), 
                          collection_name=LangchainProperties.vector_store_collection_name())