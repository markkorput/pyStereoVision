from optparse import OptionParser
# import datetime as dt,
import sys, time, json, os, logging
import cv2
import numpy as np

def create_processor(data):
  typ = data['type'] if 'type' in data else None
  
  def enhance(func):
    verbose = data['verbose'] if 'verbose' in data else False
    enabled = not ('enabled' in data and data['enabled'] == False)

    def finalfunc(frame):
      if not enabled: return frame
      if verbose: print('[fx] {}'.format(typ))
      return func(frame)
    return finalfunc

  def select(val, values, aliases=[]):
    if val in aliases:
      val = aliases.index(val)
    elif type(val) == type('') and val.isdigit():
      val = int(val)

    if type(val) == type(0):
      if val < 0 or val >= len(values):
        return values[0]
      return values[val]

    return values[0]

  def select_int(val, min=None, max=None):
    val = int(val)
    if min != None and val < min:
      return min
    if max != None and val > max:
      return max
    return val

  if typ == 'grayscale':
    def func(f):
      return cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    return enhance(func)

  if typ == 'invert':
    def func(f):
      return 255-f
    return enhance(func)

  if typ == 'blur':
    x = data['x'] if 'x' in data else 0.0
    y = data['y'] if 'y' in data else 0.0
    sx = data['sigma-x'] if 'sigma-x' in data else 10
    sy = data['sigma-y'] if 'sigma-y' in data else 10

    def func(f):
      return cv2.GaussianBlur(f, (x,y), sx,sy)
    return enhance(func)


  # if typ == 'diff'

  if typ == 'threshold':
    value = data['value'] if 'value' in data else 0
    mx = data['max'] if 'max' in data else 255

    types = {
      'CHAIN_APPROX_NONE':cv2.CHAIN_APPROX_NONE,
      'CHAIN_APPROX_SIMPLE':cv2.CHAIN_APPROX_SIMPLE,
      'CHAIN_APPROX_TC89_L1':cv2.CHAIN_APPROX_TC89_L1,
      'CHAIN_APPROX_TC89_KCOS':cv2.CHAIN_APPROX_TC89_KCOS
    }
    threshold_type = types[data['method']] if 'method' in data and data['method'] in types else None
    if threshold_type == None and 'method' in data and type(data['method']) == type(1):
      types = [cv2.CHAIN_APPROX_NONE,cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS]
      typeno = int(data['method'])
      if typeno < 0 or typeno >= len(types):
        typeno = 0
      threshold_type = types[typeno]

    def func(frame):
      ret,f = cv2.threshold(frame, value, mx, threshold_type if threshold_type else cv2.CHAIN_APPROX_NONE)
      return f
    return enhance(func)


  if typ == 'dilate':
    val = data['kernel'] if 'kernel' in data else 7
    iters = data['iterations'] if 'iterations' in data else 3
    kernel = np.ones((val, val),np.uint8)

    def func(frame):
      return cv2.dilate(frame, kernel, iterations=iters)

    return enhance(func)



  if typ == 'contours':     
    mode = select(data['mode'] if 'mode' in data else 0, [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE, cv2.RETR_FLOODFILL], aliases=['RETR_EXTERNAL', 'RETR_LIST', 'RETR_CCOMP', 'RETR_TREE', 'RETR_FLOODFILL'])
    method = select(data['method'] if 'method' in data else 0, [cv2.CHAIN_APPROX_NONE,cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS])
    drawlines = drawboxes = data['drawlines'] if 'drawlines' in data else True
    linethickness = select(data['linethickness'] if 'linethickness' in data else 1, [-1,0,1,2,3,4,5], aliases=[-1,0,1,2,3,4,5])
    drawboxes = data['drawboxes'] if 'drawboxes' in data else False
    minsize = select_int(data['minsize'] if 'minsize' in data else 0, max=4000)

    def func(frame):
      contours,hierarchy = cv2.findContours(frame, mode, method)
      if drawlines:
        frame = cv2.drawContours(frame,contours,-1,(255,255,0),linethickness)
      if drawboxes:
        for c in contours:
          bx,by,bw,bh = cv2.boundingRect(c)
          if minsize == 0 or (bw*bh) >= minsize:
            frame = cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(255,255,0),linethickness)
      return frame

    return enhance(func)

  return None

def create_processor_from_file(filepath):
  # config file exists?
  if not os.path.isfile(filepath):
    return None

  # read file content
  text = None
  with open(filepath, "r") as f:
      text = f.read()

  # parse json
  data = None
  try:
    data = json.loads(text)
  except json.decoder.JSONDecodeError as err:
    logging.warning('Could not load calibration json: \n{}'.format(err))
    return None

  # create processor for each processor in config
  processors = []
  for processor_data in data['processors']:
    p = create_processor(processor_data)
    if p:
      processors.append(p)

  # create wrapping processor which runs individual processors in sequence
  def processor_func(frame):
    f = frame
    for p in processors:
      f = p(f)
    return f

  # return wrapping processor func
  return processor_func


if __name__ == '__main__':
  # parse command line arguments
  parser = OptionParser()

  parser.add_option('-i', '--input', dest='input', default='saved-media/studio-test01-L-short.mp4')
  parser.add_option('-c', '--config-file', dest='config_file', default='data/process.json')
  # parser.add_option('-s', '--start', dest='start', default='10:15')
  # parser.add_option('-e', '--end', dest='end', default='21:45')
  # parser.add_option('-i', '--interval', dest='interval', default=10, type="float")
  # parser.add_option('-c', '--count', dest='count', default=None, type="int") # s
  # parser.add_option('-o', '--once', dest='once', action="store_true", default=False)
  # parser.add_option('-v', '--verbose', dest='verbose', action="store_true", default=False)

  opts, args = parser.parse_args()

  # print('Opening videocapture for: {}'.format(opts.input))
  iscam = opts.input.isdigit()
  capture = cv2.VideoCapture(int(opts.input) if iscam else opts.input)
  processor = create_processor_from_file(opts.config_file)

  isDone = False
  try:
    while not isDone:
      # self.lastframe = s.cap.read()[1]
      ret,frame = capture.read()
      if ret:
        # self.lastframe = frame
        processed_frame = processor(frame) if processor else frame
        cv2.imshow('processed frame', processed_frame)

      elif not iscam:
        # loop
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

      # process user input
      key = cv2.waitKey(20) & 0xFF
      if key == 27 or key == ord('q'): # escape or Q
        isDone = True

      # if key == ord(' '):
      #   isPaused = not isPaused

      # if key == ord('s'):
      #   saveframe = True

      if key == ord('r'):
        # captureRef = True
        processor = create_processor_from_file(opts.config_file)

      if isDone:
        break

  except KeyboardInterrupt:
    print('[process.py] KeyboardInterrupt, stopping.')
