
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

class Connector(ABC):
  """abstract connector for connect anything"""
  
  def __init__(self) -> None:
    self.documents: List[Document] = []
  
  def load_documents(self, uri: str):
    """load source document from uri"""
    self.documents = self.load(uri)
    
  @abstractmethod
  def load(self, uri: str):
    """load source document from uri"""
    pass
  
  def to_content(self) -> str:
    """convert document list to text content"""
    documents_content = '\n'.join(doc.page_content for doc in self.documents)
    return documents_content
  
  def to_chunk(self) -> List[Document]:
    isShort = len(self.documents[0].page_content) < 5000
    split_size = 1000 if isShort else 2000
    overlap_count = 20 if isShort else 100
    sep_char = '\n' if isShort else ''
    text_splitter = CharacterTextSplitter(
      chunk_size=split_size, chunk_overlap=overlap_count, separator=sep_char
    )
    return text_splitter.split_documents(self.documents)
  
  
    