import src.datasets as ds

if __name__ == '__main__':
  dataset = ds.WekaWeather.fromfile("resources/weka-weather.csv")
  entropy = dataset.entropy('play')
  print(dataset)
  print()

  print(f"H(Play) Entropy of the weka weather dataset: {entropy:.2f}")
  for class_ in dataset.classes('outlook'):
    class_entropy = dataset[dataset['outlook'] == class_].entropy('play')
    print(f"H(Play|Outlook={class_}): {class_entropy:.2f}")
  print()

  for label in dataset.labels():
    information_gain = dataset.information_gain(label, 'play')
    print(f"IG({label}): {information_gain:.4f}")
  print()
