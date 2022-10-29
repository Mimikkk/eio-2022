import csv
from .types import Dataset

class TitanicDatasetRow(Dataset):
  p_id: int
  p_class: int
  sex: str
  age: int
  siblings: int
  family: int
  survived: bool
  total: int
class TitanicDataset(tuple[TitanicDatasetRow, ...]):
  @classmethod
  def fromfile(cls, path: str) -> 'TitanicDataset':
    with open(path) as file:
      next(lines := csv.reader(file, delimiter=","))
      return cls(
        {
          "p_id": int(p_id),
          "p_class": int(p_class),
          "sex": sex,
          "age": int(age),
          "siblings": int(siblings),
          "family": int(family),
          "survived": bool(int(survived)),
        } for (p_id, p_class, _, sex, age, siblings, family, survived) in lines)