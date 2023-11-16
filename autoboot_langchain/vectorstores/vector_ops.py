
import abc
from typing import Optional, List, Type
from autoboot_langchain.llms.llm_factory import get_embeddings, get_chat_llm
from autoboot_langchain.connector.connector_factory import from_uri
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain, StuffDocumentsChain
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores import VectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class VectorOps(abc.ABC):
  
  def summarize(self, uri: str) -> str:
    map_template = """以下是文档的内容
    {docs}
    将之提炼出摘要:"""
    prompt = PromptTemplate.from_template(map_template)
    llm = get_chat_llm()
    
    map_chain = LLMChain(llm=llm, prompt=prompt)
    reduce_template = """以下是一组摘要:
    {doc_summaries}
    将这些内容合并为一段最终文本，并提出3个问题与回答:"""
    
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    combine_documents_chain = StuffDocumentsChain(
      llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )
    reduce_documents_chain = ReduceDocumentsChain(
      # This is final chain that is called.
      combine_documents_chain=combine_documents_chain,
      # If documents exceed context for `StuffDocumentsChain`
      collapse_documents_chain=combine_documents_chain,
      # The maximum number of tokens to group documents into.
      token_max=16000,
    )
    map_reduce_chain = MapReduceDocumentsChain(
      llm_chain=map_chain,
      reduce_documents_chain=reduce_documents_chain,
      # The variable name in the llm_chain to put the documents in
      document_variable_name="docs",
      # Return the results of the map steps in the output with 'intermediate_steps'
      return_intermediate_steps=True,
    )
    
    docs = from_uri(uri).to_chunk()
    outputs = map_reduce_chain.invoke({"input_documents": docs}, return_only_outputs=True)
    print(outputs)
  
  
  def store_index(self, uri: str) -> VectorStore:
    return self.do_store_index(self, from_uri(uri).to_chunk(), get_embeddings())
  
  async def search(self, query: str, vector_store: Optional[VectorStore] = None):
    retriever: VectorStoreRetriever
    if vector_store:
      retriever = vector_store.as_retriever(search_kwargs={"k" : 4})
    else:
      retriever = self.create_store_retriever()
    
    docs = await retriever.aget_relevant_documents(query)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(llm=get_chat_llm(),
                              chain_type="refine",
                              retriever=retriever,
                              memory=memory,
                              return_source_documents=True)
    conversationChain.run(input_documents=docs, question=query)
  
  @abc.abstractmethod
  def get_type(self) -> Type[VectorStore]:
    pass
  
  @abc.abstractmethod
  def get_connection_args(self) -> dict:
    pass
  
  @abc.abstractmethod
  def create_store_retriever(self) -> VectorStoreRetriever:
    pass
  
  @abc.abstractmethod
  def do_store_index(self, docs: List[Document], embedding: Embeddings) -> VectorStore:
    pass