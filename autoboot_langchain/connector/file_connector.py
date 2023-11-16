
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import (
  PDFMinerLoader,
  TextLoader, 
  UnstructuredImageLoader, 
  UnstructuredMarkdownLoader,
  UnstructuredWordDocumentLoader,
)
from autoboot_langchain.util.recognizer import detect_document_type
from .connector import Connector

class FileConnector(Connector):
  
  def load(self, uri: str) -> List[Document]:
    loader: BaseLoader
    file_type = detect_document_type(uri)
    if(file_type == "txt"):
      loader = TextLoader(uri, encoding='utf8')
    elif(file_type == "image"):
      loader = UnstructuredImageLoader(uri)
    elif(file_type == "md"):
      loader = UnstructuredMarkdownLoader(uri)
    elif(file_type == "doc"):
      loader = UnstructuredWordDocumentLoader(uri)
    elif(file_type == "pdf"):
      loader = PDFMinerLoader(uri)
      # resolve pdf to html with BeautifulSoup
      #soup = BeautifulSoup(loader.load()[0].page_content,'html.parser')
      #content = soup.find_all('div')
    documents = loader.load()
    return documents