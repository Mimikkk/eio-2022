from treelib import Tree

import src.datasets as ds

def find_most_informative_feature(dataset, decision: str):
  return max(dataset.labels(decision), key=lambda x: dataset.information_gain(x, decision))

def create_tree(dataset: ds.Dataset, decision: str):
  best_feature = find_most_informative_feature(dataset, decision)
  print(best_feature)
  tree = {best_feature: {}}
  for f_class in dataset.classes(best_feature):
    print(f_class)
    subset = dataset[best_feature, f_class]
    if len(subset) == 0:
      tree[best_feature][f_class] = None
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

offset = 24
decision = 'survived'
if __name__ == '__main__':
  verify(ds.WekaWeather.fromfile("resources/weka-weather.csv"), 'play')

  dataset = ds.Titanic.fromfile("resources/titanic.csv")
  dataset.omit(('p_id', 'name'), inline=True)

  mean = dataset[dataset[decision] == True]['age'].mean()
  dataset['age'] = dataset['age'].map(lambda x: x <= mean and f'<={mean:.2f}' or f'>{mean:.2f}')

  treeify(create_tree(dataset, decision)).show()
