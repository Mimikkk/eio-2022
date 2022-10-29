from typing import TypedDict, Iterable, overload
from typing import TypeVar
from ..stats import maths

class DatasetRow(TypedDict): pass

X = TypeVar("DatasetRow", bound=TypedDict)
class Dataset(tuple[X, ...]):
  def entropy(self, feature: str, /, *, base: int = 2) -> float:
    total = len(self)

    return sum(map(
      lambda count: maths.entropy(count / total, base=base),
      (self.counts(feature, c) for c in self.classes(feature))
    ))

  def information(self, feature: str, decision: str, /, *, base: int = 2) -> float:
    total = len(self)

    return sum(
      (len(subset) / total) * subset.entropy(decision, base=base)
      for subset in (self[self[feature] == value] for value in self.classes(feature))
    )

  def information_gain(self, feature: str, decision: str, /, *, base: int = 2) -> float:
    return self.entropy(decision, base=base) - self.information(feature, decision, base=base)

  def split_information(self, feature: str, /, *, base: int = 2) -> float:
    return self.entropy(feature, base=base)

  def information_gain_ratio(self, feature: str, decision: str, /, *, base: int = 2) -> float:
    return self.information_gain(feature, decision, base=base) / self.split_information(feature, base=base)

  def find_most_informative_feature(self, decision: str):
    return max(self.labels(decision), key=lambda x: self.information_gain_ratio(x, decision))

  def counts(self, label: str, l_class: str):
    return sum(1 for row in self if row[label] == l_class)

  def classes(self, label: str):
    return set(row[label] for row in self)

  def labels(self, _except: tuple[str]):
    return tuple(filter(lambda x: x not in _except, self[0]))

  def __getitem__(self, value):
    if isinstance(value, str):
      return self.Series(row[value] for row in self)
    if isinstance(value, tuple):
      it = iter(value)
      return type(self)(row for row in self if next(it))
    return super().__getitem__(value)

  def __str__(self):
    rows = '\n'.join(map(str, self))
    headers = ", ".join(map(str, self.labels([])))
    return f"Labels:\n- {headers}\nData:\n{rows}"

  class Series(tuple):
    def __eq__(self, key: str):
      return type(self)(key == x for x in self)
