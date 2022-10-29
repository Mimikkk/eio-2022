from copy import deepcopy

import src.datasets as ds

def find_most_informative_feature(dataset, decision: str):
  return max(dataset.labels(decision), key=lambda x: dataset.information_gain(x, decision))

def create_tree(dataset: ds.Dataset, decision: str):
  dataset = deepcopy(dataset)

  best_feature = find_most_informative_feature(dataset, decision)
  print(f"Best feature: {best_feature}")

  tree = {}
  for value in dataset.classes(best_feature):
    subset = dataset[best_feature, value]
    if len(subset) == 0:
      tree[value] = None
    elif len(subset.classes(decision)) == 1:
      tree[value] = subset[0][decision]
    else:
      tree[value] = create_tree(subset, decision)

  return tree

decision = 'play'
offset = 24
if __name__ == '__main__':
  dataset = ds.WekaWeather.fromfile("resources/weka-weather.csv")
  print(dataset[('outlook', 'sunny'), ('temperature', 'hot'), ('humidity', 'high')])
  # entropy = dataset.entropy(decision)
  # print(dataset)
  # print()
  #
  # print(f"H(Play) Entropy of the weka weather dataset: {entropy:.2f}")
  # for class_ in dataset.classes('outlook'):
  #   class_entropy = dataset[dataset['outlook'] == class_].entropy(decision)
  #   print(f"{f'H(Play|Outlook={class_})': <{offset}}: {class_entropy:.2f}")
  # print()
  #
  # for label in dataset.labels([decision]):
  #   information = dataset.information(label, decision)
  #   information_gain = dataset.information_gain(label, decision)
  #   split_information = dataset.split_information(label)
  #   information_gain_ratio = dataset.information_gain_ratio(label, decision)
  #   print(f"{f'I({label})': <{offset}}: {information:.4f}")
  #   print(f"{f'IG({label})': <{offset}}: {information_gain:.4f}")
  #   print(f"{f'SI({label})': <{offset}}: {split_information:.4f}")
  #   print(f"{f'IGR({label})': <{offset}}: {information_gain_ratio:.4f}")
  # print()

  label = find_most_informative_feature(dataset, decision)
  print(f"Max information gain from: {label}")
  tree = create_tree(dataset, decision)
  print(tree)
