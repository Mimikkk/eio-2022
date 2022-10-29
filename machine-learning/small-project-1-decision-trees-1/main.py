import src.datasets as ds

if __name__ == '__main__':
  dataset = ds.WekaWeather.fromfile("resources/weka-weather.csv")
  entropy = dataset.entropy('play', (1, 0))
  print(f"H(Play) Entropy of the weka weather dataset: {entropy:.2f}")
