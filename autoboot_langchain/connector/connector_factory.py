
from .connector import Connector
from .url_connector import URLConnector
from .file_connector import FileConnector


def from_uri(uri: str) -> Connector:
  connector: Connector
  if(uri.startswith("http")):
    connector = URLConnector()
  else:
    connector = FileConnector()
  connector.load_documents(uri)
  return connector