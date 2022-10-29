import src.datasets as ds

decision = 'play'
if __name__ == '__main__':
  dataset = ds.WekaWeather.fromfile("resources/weka-weather.csv")
  entropy = dataset.entropy(decision)
  print(dataset)
  print()

  print(f"H(Play) Entropy of the weka weather dataset: {entropy:.2f}")
  for class_ in dataset.classes('outlook'):
    class_entropy = dataset[dataset['outlook'] == class_].entropy(decision)
    print(f"H(Play|Outlook={class_}): {class_entropy:.2f}")
  print()

  for label in dataset.labels([decision]):
    information = dataset.information(label, decision)
    information_gain = dataset.information_gain(label, decision)
    split_information = dataset.split_information(label)
    information_gain_ratio = dataset.information_gain_ratio(label, decision)
    print(f"I({label}): {information:.4f}")
    print(f"IG({label}): {information_gain:.4f}")
    print(f"SI({label}): {split_information:.4f}")
    print(f"IGR({label}): {information_gain_ratio:.4f}")
  print()

  label = dataset.find_most_informative_feature(decision)
  print(f"Max information gain from: {label}")
