# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json
import numpy as np
from optparse import OptionParser
from datetime import datetime
from .utils.CalibrationFile import CalibrationFile

class Stream:
  def __init__(self, input, output, loop=True):
    self.id = input
    self.cap = self.createVideoCapture()
    self.writer = None
    self.loop = loop

    self.params = {
      'blur-enabled': 0,
      'blur-x': 0,
      'blur-y': 0,
      'blur-sigma-x': 0,
      'blur-sigma-y': 0,

      'canny-enabled': 0,
      'canny-threshold1': 100,
      'canny-threshold2': 200
    }

    if output:
      if output == 'auto':
        filename, ext = os.path.splitext(output)
        output = '{}-UNDISTORTED{}'.format(filename,ext)

      VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        #'mp4': cv2.VideoWriter_fourcc(*'H264'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
      }

      res = (640,360)
      fps = 24

      logging.info("Creating VideoWriter to: {}, {}fps, {}x{}px".format(output, fps, res[0], res[1]))
      self.writer = cv2.VideoWriter(output, VIDEO_TYPE[os.path.splitext(output)[1][1:]], fps, res)

    self.refFrame = None
    self.lastCapturedFrame = None
    self.lastProcessedFrame = None

  def __del__(self):
    if self.cap:
      self.cap.release()
      self.cap = None
    if self.writer:
      self.writer.release()
      self.writer = None

  def createVideoCapture(self):
    return cv2.VideoCapture(int(self.id) if self.id.isdigit() else self.id)

  def update(self, captureRef=False):
    ret, frame = self.cap.read()
    if not ret:
      if self.loop:
        logging.info("Looping {}".format(self.id))
        self.cap = self.createVideoCapture()
      return self.lastCapturedFrame

    self.lastCapturedFrame = frame
    
    if captureRef:
      self.refFrame = self.lastCapturedFrame
      logging.info('Captured ref frame for input: {}'.format(self.id))

    if type(self.refFrame) == type(None):
      # logging.info('Not ref frame for input: {}'.format(self.id))
      return self.lastCapturedFrame

    frame = cv2.absdiff(frame, self.refFrame)
    if self.params['blur-enabled'] and self.params['blur-x'] > 0 and self.params['blur-y'] > 0:
      frame = cv2.GaussianBlur(frame, (self.params['blur-x'],self.params['blur-y']), self.params['blur-sigma-x'], self.params['blur-sigma-y'])

    if self.params['canny-enabled']:
      frame = cv2.Canny(frame, self.params['canny-threshold1'], self.params['canny-threshold2'])

    if self.writer:
      self.writer.write(self.lastProcessedFrame)
    
    self.lastProcessedFrame = frame
    return frame

def createGui(streams):

  for idx, s in enumerate(streams):
    winid = 'GUI-{}'.format(s.id)
    cv2.namedWindow(winid, cv2.WINDOW_NORMAL)

    def addStreamParam(param, max, valueProc=None):
      def onValue(val):
        s.params[param] = valueProc(val) if valueProc else val
      cv2.createTrackbar(param, winid, s.params[param], max, onValue)

    blurvalues = [0,1,3,5,7,9,11,13,15,17,19]
    addStreamParam('blur-enabled', 1)
    addStreamParam('blur-x', len(blurvalues)-1, valueProc=lambda v: blurvalues[v])
    addStreamParam('blur-y', len(blurvalues)-1, valueProc=lambda v: blurvalues[v])
    addStreamParam('blur-sigma-x', 10)
    addStreamParam('blur-sigma-y', 10)
    addStreamParam('canny-enabled', 1)
    addStreamParam('canny-threshold1', 500)
    addStreamParam('canny-threshold2', 500)

    cv2.moveWindow(winid, 5, 5 + 400*idx)
    cv2.resizeWindow(winid, 500,400)
    # cv2.setWindowProperty(winid,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(winid,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)

def main(input, output=None, calibrationFilePath=None, crop=True, delay=0, verbose=False, outvideo=None):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  inputs = str(input).split(',')
  outputs = None if output == None else str(output).split(',')

  if outputs and len(inputs) != len(outputs):
    logging.warning('Number of inputs ({}: {}) doesn\'t match with number of outputs ({}: {})'.format(len(inputs), inputs, len(outputs), outputs))
    return

  # calibfile = CalibrationFile(calibrationFilePath)
  streams = []

  for idx, inp in enumerate(inputs):
    # calibdata = calibfile.getDataForVideoId(inp)
    streams.append(Stream(inp, outputs[idx] if outputs else None))

  createGui(streams)

  logging.info("Starting undistort playback, press <ESC> or 'Q' or CTRL+C to stop, <SPACE> to pause and 'S' to save a frame...")
  isPaused = False
  # saveframe = False
  captureRef = True

  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if time.time() > nextFrameTime:
          
          # isDone = update(streams, crop, frameCallback)
          for s in streams:
            frame = s.update(captureRef=captureRef)
            cv2.imshow('Input: {} - Press R to (re-)set reference frame'.format(s.id), frame)

          captureRef = False

          # saveframe = False
          if delay:
            nextFrameTime = time.time() + delay

      # process user input
      key = cv2.waitKey(20) & 0xFF
      if key == 27 or key == ord('q'): # escape or Q
        isDone = True

      if key == ord(' '):
        isPaused = not isPaused

      if key == ord('s'):
        saveframe = True

      if key == ord('r'):
        captureRef = True

      if isDone:
        break

  except KeyboardInterrupt:
    logging.info("KeyboardInterrupt, stopping")


  cv2.destroyAllWindows()



if __name__ == '__main__':
  parser = OptionParser()

  parser.add_option("-i", "--input", dest="input",
                    help="Video file to read from, (default: %default)",
                    default='saved-media/base75mm-pattern22mm-short_L.avi')

  parser.add_option("-o", "--output", dest="output", type="string",
                    help="Path to file where undistorted video should be saved",
                    default=None)

  parser.add_option("-d", "--delay", dest="delay", type="float",
                    help="Delay between each frame",
                    default=None)

  parser.add_option("-c", "--crop",
                    action="store_true", dest="crop", default=False,
                    help="Crop undistorted images when previewing calibration results")

  parser.add_option("-f", "--calibration-file", dest="calibfile", type="string",
                    help="Path calibration file",
                    default='saved-media/calibration.json')

  parser.add_option("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Verbose logging to stdout")
  

  (options, args) = parser.parse_args()

  main(input=options.input, output=options.output, calibrationFilePath=options.calibfile, crop=options.crop, delay=options.delay, verbose=options.verbose)




