from treelib import Tree

import src.datasets as ds

def create_tree(dataset: ds.Dataset, decision: str):
  def find_best_node(dataset, decision: str):
    return max(dataset.labels(decision), key=lambda x: dataset.information_gain(x, decision))

  best_feature = find_best_node(dataset, decision)
  tree = {best_feature: {}}
  for f_class in dataset.classes(best_feature):
    subset = dataset[best_feature, f_class]
    if subset == dataset: subset = subset.prune_errors(decision)

    if len(subset) == 0:
      tree[best_feature][f_class] = False
    elif len(subset.classes(decision)) == 1:
      tree[best_feature][f_class] = subset[0][decision]
    else:
      tree[best_feature][f_class] = create_tree(subset, decision)

  return tree

def treeify(data: dict):
  prefix = lambda a, b: f"{(a and f'{a},') or ''}{b}"
  def visualizer(data: dict, p_id=""):
    for (index, (key, value)) in enumerate(data.items()):
      n_id = prefix(p_id, index)
      if isinstance(value, dict):
        yield p_id, n_id, f"{str(key).capitalize()}"
        yield from visualizer(value, n_id)
      else:
        yield p_id, n_id, f"{str(key).capitalize()} -> {value}"

  tree = Tree()
  for (parent, node, value) in visualizer(data):
    if not parent:
      tree.create_node(value, node)
    else:
      tree.create_node(str(value), node, parent=parent)
  return tree

def verify(dataset: ds.Dataset, decision: str):
  entropy = dataset.entropy(decision)
  print("Student: Daniel Zdancewicz")
  print("Index: 145317")
  print()
  print(dataset)
  print()

  print(f"H(Play) Entropy of the weka weather dataset: {entropy:.2f}")
  for class_ in dataset.classes('outlook'):
    class_entropy = dataset[dataset['outlook'] == class_].entropy(decision)
    print(f"{f'H(Play|Outlook={class_})': <{offset}}: {class_entropy:.2f}")
  print()

  for label in dataset.labels([decision]):
    information = dataset.information(label, decision)
    information_gain = dataset.information_gain(label, decision)
    split_information = dataset.split_information(label)
    information_gain_ratio = dataset.information_gain_ratio(label, decision)
    print(f"{f'I({label})': <{offset}}: {information:.4f}")
    print(f"{f'IG({label})': <{offset}}: {information_gain:.4f}")
    print(f"{f'SI({label})': <{offset}}: {split_information:.4f}")
    print(f"{f'IGR({label})': <{offset}}: {information_gain_ratio:.4f}")
  print()
  treeify(create_tree(dataset, decision)).show()

def predict(tree: dict, data: dict):
  key: str
  if isinstance(tree, bool):
    return tree
  for (key, value) in tree.items():
    if isinstance(value, dict):
      if key in data:
        return predict(value[data[key]], data)
      else:
        return value
offset = 24
decision = 'survived'
if __name__ == '__main__':
  verify(ds.WekaWeather.fromfile("resources/weka-weather.csv"), 'play')

  (train_set, test_set) = ds.Titanic \
    .fromfile("resources/titanic.csv") \
    .omit(('p_id', 'name'), inline=True) \
    .split(0.8)

  age_mean = train_set[train_set[decision] == True]['age'].mean()
  train_set['age'] = train_set['age'].map(lambda x: x <= age_mean and f'<={age_mean:.2f}' or f'>{age_mean:.2f}')
  print(train_set)

  treeify(tree := create_tree(train_set, decision)).show()
  accuracy = sum(1 for row in test_set if predict(tree, row) == row[decision]) / len(test_set)
  print(f"Accuracy: {accuracy:.2f}")
