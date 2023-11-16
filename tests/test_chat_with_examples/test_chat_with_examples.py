from autoboot import AutoBoot, AutoBootConfig
from autoboot_langchain.llms import llm_factory
from autoboot_langchain.llms import llm_executor


def test_chat_with_examples():
  autoboot = AutoBoot(config=AutoBootConfig(config_dir="./tests/test_chat_with_examples"))
  autoboot.run()
  
  # 根据提示与回答例子回答问题
  examples = [
    {
      "question": "你是谁？",
      "answer": "我是一个人工智能机器人Touchat AI。"
    },
    {
      "question": "你是GPT吗？",
      "answer": "不，我是Touchat AI。"
    },
    {
      "question": "你是哪家公司开发的？",
      "answer": "我是Touchat AI，由游链网络科技开发。"
    }
  ]
  prompt = llm_executor.prompt_with_examples(examples=examples)
  llm = llm_factory.get_llm()
  print(llm(prompt.format(input="你是谁？")))