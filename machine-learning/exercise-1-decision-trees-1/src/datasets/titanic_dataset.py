import csv
from .dataset import DatasetRow, Dataset

class Row(DatasetRow):
  p_id: int
  p_class: int
  name: str
  sex: str
  age: int
  siblings: int
  family: int
  survived: bool

class Titanic(Dataset[Row]):
  @classmethod
  def fromfile(cls, path: str, *, simplify: bool = False) -> 'Titanic':
    with open(path) as file:
      next(lines := csv.reader(file, delimiter=","))
      return cls(
        {
          "p_id": int(p_id),
          'p_class': int(p_class),
          "name": name,
          'sex': sex,
          'age': cls._simplify_age(int(age)) if simplify else int(age),
          'siblings': int(siblings),
          'family': int(family),
          'survived': bool(int(survived)),
        } for (p_id, p_class, name, sex, age, siblings, family, survived) in lines)

  @staticmethod
  def _simplify_age(age: int) -> str:
    match age:
      case n if n <= 20: return 'young'
      case n if n <= 40: return 'middle'
      case _: return 'old'
