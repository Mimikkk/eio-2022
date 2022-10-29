from typing import TypedDict, Iterable, overload
from typing import TypeVar
from ..stats import maths

class DatasetRow(TypedDict): pass

X = TypeVar("DatasetRow", bound=TypedDict)
class Dataset(tuple[X, ...]):
  def entropy(self, label: str, l_classes: Iterable[str] = None, /, *, base: int = 2) -> float:
    if l_classes is None: l_classes = self.classes(label)

    total = len(self)
    return sum(map(
      lambda count: maths.entropy(count / total, base=base),
      filter(lambda count: count != 0, (self.counts(label, c) for c in l_classes))
    ))

  def information(self, feature: str, decision: str):
    total = len(self)

    return sum(
      (len(subset) / total) * subset.entropy(decision)
      for subset in (self[self[feature] == value] for value in self.classes(feature))
    )

  def information_gain(self, feature: str, decision: str):
    return self.entropy(decision) - self.information(feature, decision)

  def split_information(self, feature: str):
    total = len(self)

    return sum(map(
      lambda count: maths.entropy(count / total, base=2),
      filter(lambda count: count != 0, (self.counts(feature, l_class) for l_class in self.classes(feature)))
    ))

  def information_gain_ratio(self, feature: str, decision: str):
    return self.information_gain(feature, decision) / self.split_information(feature)

  def find_most_informative_feature(self, decision: str):
    return max(self.labels(decision), key=lambda x: self.information_gain(x, decision))

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
