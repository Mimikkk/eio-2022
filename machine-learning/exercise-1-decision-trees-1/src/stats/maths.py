import math
from typing import Iterable

def entropy(probability: float | Iterable[float] | dict[str, float], *, base: int) -> float:
  if isinstance(probability, dict):
    return entropy(probability.values(), base=base)
  if isinstance(probability, Iterable):
    return sum(map(lambda x: entropy(x, base=base), probability))
  if probability == 0: return 0
  return -probability * math.log(probability, base)
