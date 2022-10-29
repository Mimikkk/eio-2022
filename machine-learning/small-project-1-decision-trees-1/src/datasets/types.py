from typing import TypedDict, Iterable
from typing import TypeVar
from ..stats import maths

class DatasetRow(TypedDict): pass

X = TypeVar("DatasetRow", bound=TypedDict)
class Dataset(tuple[X, ...]):
  def entropy(self, label: str, l_classes: Iterable[str] = None, /, *, base: int = 2) -> float:
    if l_classes is None: l_classes = self.classes(label)

    total = len(self)
    return sum(map(
      lambda x: maths.entropy(x / total, base=base),
      filter(lambda x: x != 0, (self.counts(label, c) for c in l_classes))
    ))

  def counts(self, label: str, l_class: str):
    return sum(1 for row in self if row[label] == l_class)

  def classes(self, label: str):
    return set(row[label] for row in self)
