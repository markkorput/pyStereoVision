import os,logging,json
import numpy as np

def loadData(filepath, fallback=None):
  if not os.path.isfile(filepath):
    return fallback

  text = None
  with open(filepath, "r") as f:
      text = f.read()

  data = fallback
  try:
    data = json.loads(text)
  except json.decoder.JSONDecodeError as err:
    logging.warning('Failed to load calibration json file ({}), exception: \n{}'.format(filepath, err))
    return fallback

  return data

def saveData(filepath, data):
  with open(filepath, "w") as f:
    # f.write(json.dumps(json_data))
    # json.dump(data, f)
    json.dump(data, f)

  logging.debug('Wrote calibration data to: {}'.format(filepath))

def getData(data, pathlist, fallback=None):
  cursor = data

  for item in pathlist:
    if not cursor or not item in cursor:
      return fallback
    cursor = cursor[item]

  return cursor

def setData(data, pathlist, itemdata):
  cursor = data

  for cur in pathlist[:-1]:
    if not cursor or not cur in cursor:
      return data
 
  if not cursor:
    return data
  
  cursor[pathlist[-1]] = itemdata
  return data

def videoDataFromSerializable(data):
  return [
    data[0],
    np.array(data[1]),
    np.array(data[2]),
    list(map(lambda i: np.array(i), data[3])),
    list(map(lambda i: np.array(i), data[4]))
  ]

def videoDataToSerializable(data):
  return [
    data[0],
    data[1].tolist(),
    data[2].tolist(),
    list(map(lambda i: i.tolist(), data[3])),
    list(map(lambda i: i.tolist(), data[4]))
  ]

class CalibrationFile:
  def __init__(self, filepath):
    self.filepath = filepath
    self.data = None

  def get(self, pathlist):
    if not self.data:
      self.data = loadData(self.filepath, fallback={})
    return getData(self.data, pathlist)

  def getDataForVideoId(self, videoId, fallback=None):
    if not self.data:
      self.data = loadData(self.filepath, fallback={})
    serializable = getData(self.data, ['sv', 'calibration_data', videoId])
    return videoDataFromSerializable(serializable) if serializable else fallback

  def setDataForVideoId(self, videoId, data, save=True):
    if not self.data:
      self.data = loadData(self.filepath, fallback={})
    self.data = setData(self.data, ['sv', 'calibration_data', videoId], data)
    if save:
      saveData(self.filepath, self.data)
