from typing import TypedDict
from typing import TypeVar
from ..stats import maths

class DatasetRow(TypedDict): pass

X = TypeVar("DatasetRow", bound=TypedDict)
class Dataset(tuple[X, ...]):
  def counts(self, label: str, l_class: str):
    return sum(1 for row in self if row[label] == l_class)

  def entropy(self, label: str, classes: list[str], /, *, base: int = 2) -> float:
    total = len(self)
    return sum(map(
      lambda x: maths.entropy(x / total, base=base),
      filter(lambda x: x != 0, (self.counts(label, c) for c in classes))
    ))
