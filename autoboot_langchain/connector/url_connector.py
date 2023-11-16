
from typing import List, Dict, Union
import requests
import textwrap
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from .connector import Connector


class URLConnector(Connector):
  
  def load(self, uri: str) -> List[Document]:
    # see also WebBaseLoader and BSHTMLLoader
    header = {
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
      "Accept-Language": "zh-cn,zh;q=0.5",
      "Upgrade-Insecure-Requests": "1"
    }
    res = requests.get(url=uri, headers=header)
    res.encoding = "utf-8"
    if res.status_code != 200:
      raise Exception("Invalid status code!", res.status_code)
    soup = BeautifulSoup(res.text, features="lxml")
    text = textwrap.dedent(soup.get_text().strip().replace("\n\n", "\n"))
    title: str = ""
    if soup.title:
      title = str(soup.title.string)
    if title == str(None):
      title = text[:text.find("\n")]
    metadata: Dict[str, Union[str, None]] = {
        "source": uri,
        "title": title,
    }
    return [Document(page_content=text, metadata=metadata)]