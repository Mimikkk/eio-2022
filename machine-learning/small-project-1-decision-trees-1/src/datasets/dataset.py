from copy import deepcopy
from functools import reduce
from typing import TypedDict
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

  def counts(self, feature: str, f_class: str):
    return sum(1 for row in self if row[feature] == f_class)

  def classes(self, feature: str):
    return set(row[feature] for row in self)

  def labels(self, _except: tuple[str]):
    return tuple(filter(lambda x: x not in _except, self[0]))

  def omit(self, other, /, *, inline: bool = False):
    if not inline: self = deepcopy(self)
    if isinstance(other, tuple | list):
      return reduce(lambda ds, feat: ds.omit(feat, inline=inline), other, self)
    for row in self: del row[other]
    return self

  def pick(self, other, /, *, inline: bool = False):
    if inline:
      return self.omit(self.labels(other), inline=inline)

    if isinstance(other, tuple | list):
      return type(self)(
        {
          key: row[key]
          for key in filter(lambda x: x in other, row)
        } for row in self)

    return type(self)({other: row[other] for row in self})

  def __getitem__(self, value):
    if isinstance(value, str):
      return self.Series(row[value] for row in self)
    if isinstance(value, self.Series):
      it = iter(value)
      return type(self)(row for row in self if next(it))
    if isinstance(value, tuple):
      if isinstance(value[0], tuple):
        return reduce(lambda acc, pair: acc[pair], value, self)
      [first, rest] = value
      return self[self[first] == rest]

    return super().__getitem__(value)

  def __str__(self):
    rows = '\n'.join(map(str, self))
    headers = ", ".join(map(str, self.labels([])))
    return f"Labels:\n- {headers}\nData:\n{rows}"

  def __setitem__(self, key, value):
    if isinstance(value, self.Series):
      it = iter(value)
      for row in self: row[key] = next(it)
      return self
    raise NotImplementedError

  class Series(tuple):
    def __eq__(self, key: str):
      return type(self)(key == x for x in self)

    def __ne__(self, key: str):
      return - (self == key)

    def __neg__(self):
      return type(self)(not x for x in self)

    def normalize(self):
      max_ = max(self)
      min_ = min(self)
      return type(self)((x - min_) / (max_ - min_) for x in self)

    def mean(self):
      return sum(self) / len(self)

    def map(self, func):
      return type(self)(map(func, self))
