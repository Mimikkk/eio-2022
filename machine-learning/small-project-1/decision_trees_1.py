from collections import defaultdict
from itertools import pairwise, tee
import math
import string
import csv
from typing import TypedDict, Iterable

class TitanicDatasetRow(TypedDict):
  p_id: int
  p_class: int
  sex: str
  age: int
  siblings: int
  family: int
  survived: bool
TitanicDataset = tuple[TitanicDatasetRow, ...]

def readfile(path: string) -> TitanicDataset:
  with open(path) as file:
    next(lines := csv.reader(file, delimiter=","))
    return tuple(
      {
        "p_id": int(p_id),
        "p_class": int(p_class),
        "sex": sex,
        "age": int(age),
        "siblings": int(siblings),
        "family": int(family),
        "survived": bool(int(survived)),
      } for (p_id, p_class, _, sex, age, siblings, family, survived) in lines)

def calculate_entropy(variable: Iterable[float], /, *, base: int = 2):
  return -sum(x * math.log(x, base) if x else 0 for x in variable)

def calculate_conditional_entropy(*variables: Iterable[float], base: int = 2):
  it = iter(variables)
  return calculate_entropy(next(it, tuple()), base=base) \
         - sum(map(lambda x: calculate_entropy(x, base=base), it))

def calculate_information_gain(*variables: Iterable, base: int = 2):
  it = iter(variables)
  return calculate_entropy(next(it, tuple()), base=base) \
         - sum(map(lambda x: calculate_conditional_entropy(*x, base=base), it))

def calculate_information_gain_ratio(*variables: Iterable, base: int = 2):
  [first_it, second_it] = tee(iter(variables))

  return calculate_information_gain(*first_it, base=base) / calculate_entropy(next(second_it, tuple()), base=base)

def verify_math():
  offset = 10
  try:
    print("Math verification.")
    # X\Y  0   1
    #  0  1/4 1/4
    #  1  1/2  0
    pxy = (0.25, 0.25, 0.5)
    px = (0.5, 0.5)
    py = (0.75, 0.25)
    px_y = (pxy, py)
    py_x = (pxy, px)

    h_x = calculate_entropy(px)
    print(f"{'H(X)': <{offset}}: Entropy of X: {h_x:.2f}")
    assert round(h_x, 2) == 1.00, f"H(X) != 1.00. Invalid value: {h_x}"

    h_y = calculate_entropy(py)
    print(f"{'H(Y)': <{offset}}: Entropy of X: {h_y:.2f}")
    assert round(h_y, 2) == 0.81, f"H(X) != 0.81. Invalid value: {h_y}"

    h_xy = calculate_entropy(pxy)
    print(f"{'H(X,Y)': <{offset}}: Entropy of X: {h_xy:.2f}")
    assert round(h_xy, 2) == 1.50, f"H(X) != 1.50. Invalid value: {h_xy}"

    h_x_y = calculate_conditional_entropy(*px_y)
    print(f"{'H(X|Y)': <{offset}}: Conditional entropy of X given Y: {h_x_y:.2f}")
    assert round(h_x_y, 2) == 0.69, f"H(X|Y) != 0.69. Invalid value: {h_x_y}"

    h_y_x = calculate_conditional_entropy(*py_x)
    print(f"{'H(Y|X)': <{offset}}: Conditional entropy of Y given X: {h_y_x:.2f}")
    assert round(h_y_x, 2) == 0.50, f"H(Y|X) != 0.50. Invalid value: {h_y_x}"
    assert round(h_y_x, 2) != round(h_x_y, 2), "Conditional entropy should not be symmetric."

    ig_x_y = calculate_information_gain(px, px_y)
    print(f"{'I(X;Y)': <{offset}}: Information gain of X given Y: {ig_x_y:.2f}")
    assert round(ig_x_y, 2) == 0.31, f"IG(X;Y) != 0.31. Invalid value: {ig_x_y}"

    ig_y_x = calculate_information_gain(py, py_x)
    print(f"{'I(Y;X)': <{offset}}: Information gain of Y given X: {ig_y_x:.2f}")
    assert round(ig_y_x, 2) == 0.31, f"IG(Y;X) != 0.31. Invalid value: {ig_y_x}"
    assert ig_x_y == ig_x_y, "Information gain should be symmetric."

    ig_x_x = calculate_information_gain(px)
    print(f"{'I(X;X)': <{offset}}: Information gain of Y given X: {ig_x_x:.2f}")
    assert round(ig_x_x, 2) == round(h_x, 2), f"IG(X;X) != H(X). Invalid value: {ig_x_x}"

    ig_y_y = calculate_information_gain(py)
    print(f"{'I(Y;Y)': <{offset}}: Information gain of Y given Y: {ig_y_y:.2f}")
    assert round(ig_y_y, 2) == round(h_y, 2), f"IG(X;X) != H(Y). Invalid value: {ig_y_y}"

    igr_x_y = calculate_information_gain_ratio(px, px_y)
    print(f"{'IGR(X;Y)': <{offset}}: Information gain ratio of X given Y: {igr_x_y:.2f}")
    assert round(igr_x_y, 2) == 0.31, f"IGR(X;Y) != 0.31. Invalid value: {igr_x_y}"

    igr_y_x = calculate_information_gain_ratio(py, py_x)
    print(f"{'IGR(Y;X)': <{offset}}: Information gain ratio of Y given X: {igr_y_x:.2f}")
    assert round(igr_y_x, 2) == 0.38, f"IGR(Y;X) != 0.38. Invalid value: {igr_y_x}"

    print("Math is valid.")
  except AssertionError as e:
    print(f"Math is invalid: {e}.")

def verify_weka_weather():
  offset = 16
  try:
    print("Weka weather verification.")
    # Outlook  Temp.  Hum.   Wind  Result
    # sunny    hot    high     0     0
    # sunny    hot    high     1     0
    # overcast hot    high     0     1
    # rainy    mild   high     0     1
    # rainy    cool   normal   0     1
    # rainy    cool   normal   1     0
    # overcast cool   normal   1     1
    # sunny    mild   high     0     0
    # sunny    cool   normal   0     1
    # rainy    mild   normal   0     1
    # sunny    mild   normal   1     1
    # overcast mild   high     1     1
    # overcast hot    normal   0     1
    # rainy    mild   high     1     0
    # [outlook]
    #         Yes | No | Cum
    # sunny    2  | 3  |  5
    # overcast 4  | 0  |  4
    # rainy    3  | 2  |  5
    #          9  | 5  | 14
    print("Entropy: Outlook")
    p_sunny = (2 / 5, 3 / 5)
    p_overcast = (4 / 4, 0 / 4)
    p_rainy = (3 / 5, 2 / 5)
    p_outlook = (9 / 14, 5 / 14)
    h_sunny = calculate_entropy(p_sunny)
    print(f"{'H(Sunny)': <{offset}}: Entropy of Outlook conditioned on Sunny: {h_sunny:.2f}")
    assert round(h_sunny, 3) == 0.971, f"H(Sunny) != 0.971. Invalid value: {h_sunny}"

    h_overcast = calculate_entropy(p_overcast)
    print(f"{'H(Overcast)': <{offset}}: Entropy of Outlook conditioned on Overcast: {h_overcast:.2f}")
    assert round(h_overcast, 3) == 0.000, f"H(Overcast) != 0.971. Invalid value: {h_overcast}"

    h_rainy = calculate_entropy(p_rainy)
    print(f"{'H(Rainy)': <{offset}}: Entropy of Outlook conditioned on Rainy: {h_rainy:.2f}")
    assert round(h_rainy, 3) == 0.971, f"H(Rainy) != 0.971. Invalid value: {h_rainy}"

    h_outlook = calculate_entropy(p_outlook)
    print(f"{'H(Outlook)': <{offset}}: Entropy of Outlook: {h_outlook:.2f}")
    assert round(h_outlook, 3) == 0.940, f"H(Outlook) != 0.940. Invalid value: {h_outlook}"

    expected_info = h_sunny * (5 / 14) + h_overcast * (4 / 14) + h_rainy * (5 / 14)
    print(f"{'I(Outlook)': <{offset}}: Expected Information {expected_info:.2f}")

    ig_outlook = h_outlook - expected_info
    print(f"{'IG(Outlook)': <{offset}}: Information gain of Outlook: {ig_outlook:.2f}")
    print("Weka Example is sad im sad")
  except AssertionError as e:
    print(f"Weka Example is invalid: {e}.")


if __name__ == '__main__':
  dataset = readfile("resources/titanic-dataset.csv")
  verify_weka_weather()
  # verify_math()
