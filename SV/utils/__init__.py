import cv2, json, logging


def isNone(var):
  '''
  utility method to be used with variables that might carry nparray values, which are a bit of a pain when comparing to None
  '''
  return type(var) == type(None)

def addParamTrackbar(winid, params, param, max=None, initialValue=None, valueProc=None, values=None, factor=None, readProc=None, onChange=None, controlId=None, default=None):
  '''
  Convenience method for param manipulating trackbars

  Args:
    winid (str): opencv window ID
    params (dict): a dict (with string-based keys) containing the params
    max (int): maximum value
    initialValue (int): initialValue for the trackbar
    valueProc (method): a value pre-processor method that takes the trackbar value and return the value to be applied to the param
    readProc (method): a value pre-processor method that takes the param value and return trackbar value for the initial value
    values (list): a list of possible values. When specified, this will overwrite the max, valueProx and initialValue values
    factor (float): a multiplier/division value
  '''

  if isNone(default):
    if param in params:
      default = params[param] 
    else:
      default = 0

  if factor:
    max = factor
    valueProc = lambda v: float(v) / factor
    readProc = lambda v: int(v * factor)

  if values:
    max = len(values)-1
    valueProc = lambda v: values[v]
    if initialValue == None:
      initialValue = values.index(default)
      # print("{} int val: {} for def: {} in values: {}".format(param, initialValue, default, values))

  if not readProc:
    readProc = lambda v: int(v)

  def onValue(val):
    params[param] = valueProc(val) if valueProc else val
    if onChange:
      onChange()

  val = initialValue if initialValue != None else readProc(params[param] if param in params else default)
  cv2.createTrackbar(controlId if controlId else param, winid, val, max, onValue)


def createParamsGuiWin(winid, params, file=None, load=None, save=None):
  class Builder:
    def __init__(self, winid, params, file=None, load=None, save=None):
      self.winid = winid
      self.params = params
      self.file = file
      self.load = load
      self.save = save

      cv2.namedWindow(self.winid)
      # cv2.moveWindow(self.winid, 5, 5)
      # cv2.resizeWindow(self.winid, 500,400)

    # def __del__(self):
    #   if self.file and self.save != False:
    #     self._save()

    def __enter__(self):
      if self.file and self.load != False:
        self._load()  

      return self

    def __exit__(self, type, value, traceback):
      pass

    def _load(self):
      # save params to file
      logging.info('Loading params for gui win {} to file: {}'.format(self.winid, self.file))
      json_data = {}
      with open(file, 'r') as f:
        json_data = json.load(f)
      self.params.update(json_data)

    # def _save(self):
    #   # save params to file
    #   logging.info('Writing params for gui win {} to file: {}'.format(self.winid, self.file))
    #   with open(file, 'w') as f:
    #     json.dump(self.params, f)

    def add(self, paramname, *args, **kwargs):
      return addParamTrackbar(self.winid, self.params, paramname, *args, **kwargs)

  # create and return Builder instance 
  return Builder(winid, params, file, load, save)