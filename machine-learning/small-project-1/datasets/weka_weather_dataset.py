import csv
from .types import Dataset

class WekaWeatherDatasetRow(Dataset):
  outlook: str
  temperature: str
  humidity: str
  windy: bool
  play: bool
class WekaWeatherDataset(tuple[WekaWeatherDatasetRow, ...]):
  @classmethod
  def fromfile(cls, path: str) -> 'WekaWeatherDataset':
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
