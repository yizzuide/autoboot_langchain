from autoboot import ApplicationProperties, AutoBoot, AutoBootConfig
from autoboot_langchain.vectorstores import vector_ops_factory

def test_sum():
  autoboot = AutoBoot(config=AutoBootConfig(config_dir="./tests/test_doc_summarize"))
  autoboot.run()
  
  print(ApplicationProperties.app_name())
  
  vector_ops = vector_ops_factory.get_vector_ops()
  vector_ops.summarize("https://mp.weixin.qq.com/s/BIYp9DNd_9sw5O2daiHmlA")