
from typing import List, Type
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore, Qdrant
from langchain.vectorstores.base import VectorStoreRetriever
from autoboot.annotation import component
from autoboot_langchain.langchain_properties import LangchainProperties
from .vector_ops import VectorOps


class QdrantOps(VectorOps):
  
  def get_type(self) -> Type[VectorStore]:
    return Qdrant
  
  def get_connection_args(self) -> dict:
    return {
      "host": LangchainProperties.vector_store_connection_host(),
      "port": LangchainProperties.vector_store_connection_port()
    }
  
  def do_store_index(self, docs: List[Document], embedding: Embeddings) -> VectorStore:
    connection_args = self.get_connection_args()
    return Qdrant.from_documents(
      docs,
      embedding,
      host=connection_args["host"],
      port=connection_args["port"],
      collection_name=LangchainProperties.vector_store_collection_name()
    )
  
  @component(name="qdrant_store_retriever")
  def create_store_retriever(self) -> VectorStoreRetriever:
    connection_args = self.get_connection_args()
    return Qdrant.as_retriever(host=connection_args["host"],
                          port=connection_args["port"],
                          collection_name=LangchainProperties.vector_store_collection_name())
  