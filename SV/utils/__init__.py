import cv2


def addParamTrackbar(winid, params, param, max=None, initialValue=None, valueProc=None, values=None, factor=None, readProc=None):
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

  if factor:
    max = factor
    valueProc = lambda v: float(v) / factor
    readProc = lambda v: int(v * factor)

  if values:
    max = len(values)-1
    valueProc = lambda v: values[v]
    initialValue = values.index(params[param])

  if not readProc:
    readProc = lambda v: int(v)

  def onValue(val):
    params[param] = valueProc(val) if valueProc else val
  cv2.createTrackbar(param, winid, initialValue if initialValue != None else readProc(params[param]), max, onValue)
