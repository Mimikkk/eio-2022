import csv

from .dataset import Dataset, DatasetRow

class Row(DatasetRow):
  outlook: str
  temperature: str
  humidity: str
  windy: bool
  play: bool
class WekaWeather(Dataset[Row]):
  @classmethod
  def fromfile(cls, path: str) -> 'WekaWeather':
    with open(path) as file:
      next(lines := csv.reader(file, delimiter=","))
      return cls(
        {
          "outlook": outlook,
          "temperature": temperature,
          "humidity": humidity,
          "windy": True if windy == "TRUE" else False,
          "play": True if play == "yes" else False,
        } for (outlook, temperature, humidity, windy, play) in lines)
