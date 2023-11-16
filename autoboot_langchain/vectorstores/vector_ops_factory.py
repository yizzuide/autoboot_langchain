
from autoboot.annotation import component
from autoboot_langchain.langchain_properties import LangchainProperties
from .vector_ops import VectorOps
from .milvus_ops import MilvusOps
from .qdrant_ops import QdrantOps


@component(name="vector_ops")
def get_vector_ops() -> VectorOps:
  vector_store_type = LangchainProperties.vector_store_type()
  if vector_store_type == "milvus":
    return MilvusOps()
  elif vector_store_type == "qdrant":
    return QdrantOps()