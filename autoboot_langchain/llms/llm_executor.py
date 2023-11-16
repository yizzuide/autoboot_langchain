
from typing import List
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from autoboot_langchain.langchain_properties import LangchainProperties
from autoboot_langchain.vectorstores.vector_ops_factory import get_vector_ops
from autoboot_langchain.llms.llm_factory import get_embeddings

def prompt_with_examples(examples: List[dict], embedding_in_store: bool=False):
  """针对私有领域样例数据向量化的提示模板"""
  
  example_prompt = PromptTemplate(input_variables=["question", "answer"], 
                                  template="Question: {question}\n{answer}")
  #print(example_prompt.format(**examples[0]))
  
  # 先将样例数据嵌入到向量库
  if embedding_in_store:
    vector_ops = get_vector_ops()
    example_selector = SemanticSimilarityExampleSelector.from_examples(examples,
                                  get_embeddings(),
                                  vector_ops.get_type(),
                                  k=1, # K近邻算法（KNN)
                                  connection_args=vector_ops.get_connection_args(),
                                  collection_name=LangchainProperties.vector_store_collection_name())
    prompt = FewShotPromptTemplate(example_selector=example_selector,
                                  example_prompt=example_prompt,
                                  prefix="如果提问与下面例子相似，请以下内容进行推断:",
                                  suffix="Question: {input}",
                                  input_variables=['input'])
    return prompt
  
  # 直接使用样例数据
  prompt = FewShotPromptTemplate(examples=examples,
                                  example_prompt=example_prompt,
                                  prefix="如果提问与下面例子相似，请以下内容进行推断:",
                                  suffix="Question: {input}",
                                  input_variables=['input'])
  return prompt
