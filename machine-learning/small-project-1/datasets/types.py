from typing import TypedDict
from typing import TypeVar

class DatasetRow(TypedDict): pass

X = TypeVar("DatasetRow", bound=TypedDict)
class Dataset(tuple[X, ...]):
  def counts(self, label: str, l_class: str):
    return sum(1 for row in self if row[label] == l_class)
