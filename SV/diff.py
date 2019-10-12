# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json
import numpy as np
from optparse import OptionParser
from datetime import datetime
from .utils.CalibrationFile import CalibrationFile

class Stream:
  def __init__(self, input, output, calibdata, loop=True):
    self.id = input
    self.cap = self.createVideoCapture()

    self.calibrationdata = calibdata
    self.done = False
    self.writer = None
    self.loop = loop

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

  def update(self, captureRef=False, gaussianBlur=None):
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
    if gaussianBlur:
      frame = cv2.GaussianBlur(frame, *gaussianBlur) # (3, 3), 0)
    # edges = cv2.Canny(diff, 100, 200)


    if self.writer:
      self.writer.write(self.lastProcessedFrame)
    
    self.lastProcessedFrame = frame
    return frame

def get_undistort(img, calibdata, crop=True):
  ret, mtx, dist, rvecs, tvecs = calibdata
  h,  w = img.shape[:2]  
  # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

  # undistort
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

  # crop the image
  if crop:
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
  return dst


def main(input, output=None, calibrationFilePath=None, crop=True, delay=0, verbose=False, outvideo=None):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  inputs = str(input).split(',')
  outputs = None if output == None else str(output).split(',')

  if outputs and len(inputs) != len(outputs):
    logging.warning('Number of inputs ({}: {}) doesn\'t match with number of outputs ({}: {})'.format(len(inputs), inputs, len(outputs), outputs))
    return

  calibfile = CalibrationFile(calibrationFilePath)
  streams = []

  for idx, inp in enumerate(inputs):
    calibdata = calibfile.getDataForVideoId(inp)
    streams.append(Stream(inp, outputs[idx] if outputs else None, calibdata))

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
            frame = s.update(captureRef=captureRef, gaussianBlur=None) #((3,3),0))
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




