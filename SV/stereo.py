# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json, math
import numpy as np
from optparse import OptionParser
from datetime import datetime
from .utils.CalibrationFile import CalibrationFile

DEFAULTS = {
  'invideos': ['saved-media/base75mm-pattern22mm-short_L-UNDISTORTED.avi','saved-media/base75mm-pattern22mm-short_R-UNDISTORTED.avi'],
  'delay': None,
  'calibfile': None, #'saved-media/calibration.json',
  'crop': False,
  'loop': True
}

class Stream:
  def __init__(self, vid_path, calibdata):
    self.id = vid_path
    self.calibrationdata = calibdata
    self.init()

  def init(self):
    self.cap = cv2.VideoCapture(self.id)
    self.done = False

  def __del__(self):
    if self.cap:
      self.cap.release()
      self.cap = None

class Computer:
  def __init__(self, *args):
    self.set(*args)

  def set(self, minDisp, numDisparities, blockSize,p1,p2,disp12MaxDiff,uniquenessRatio,speckleWindowSize,speckleRange):
    logging.info('Initializing stereo computer with values: {}'.format(
      (numDisparities, blockSize, minDisp,p1,p2,disp12MaxDiff,uniquenessRatio,speckleWindowSize,speckleRange)))

    self.stereo = cv2.StereoSGBM_create(
      minDisparity=minDisp,
      numDisparities=numDisparities,
      blockSize=blockSize,
      P1=p1,
      P2=p2,
      disp12MaxDiff=disp12MaxDiff,
      uniquenessRatio = uniquenessRatio,
      speckleWindowSize = speckleWindowSize,
      speckleRange = speckleRange)

    # self.stereo = cv2.StereoSGBM_create(minDisparity=minDisp, blockSize=blockSize)


  def compute(self, *args):
    return self.stereo.compute(*args)


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

def update(streams, computer, crop, showinput, disparityFrameCallback):
  frames = []

  # for s in streams:
  for s in streams:
    (retval, frame) = s.cap.read()
    if retval:
      undistorted_image = get_undistort(frame, s.calibrationdata, crop=crop) if s.calibrationdata else frame
      if showinput:
        cv2.imshow("{} - UNDISTORTED".format(s.id), undistorted_image)
      #if s.writer:
      #  s.writer.write(undistorted_image)
      frames.append(undistorted_image)
    else:
      s.done = True

  if len(frames) == 2:
    logging.debug('Computing disparity...')
    disp = computer.compute(frames[0], frames[1]).astype(np.float32) / 16.0

    # disp = getDisparity(pair[0], pair[1])
    # cv2.imshow("DISPARITY", disp)
    # cv2.imshow('DISPARITY', (disp-min_disp)/num_disp)
    disparityFrameCallback(disp)


  return len(list(filter(lambda s: s.done == True, streams))) > 0

def main(video_paths, calibrationFilePath=None, crop=True, delay=0, verbose=False, outvideo=None, loop=False, showinput=False):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  # input streams
  streams = []
  calibfile = CalibrationFile(calibrationFilePath) if calibrationFilePath else None

  for vid_path in video_paths:
    calibdata = calibfile.getDataForVideoId(vid_path) if calibfile else None
    streams.append(Stream(vid_path, calibdata))

  # disparity output writer
  disparityWriter = None
  if outvideo:
    VIDEO_TYPE = {
      'avi': cv2.VideoWriter_fourcc(*'XVID'),
      #'mp4': cv2.VideoWriter_fourcc(*'H264'),
      'mp4': cv2.VideoWriter_fourcc(*'XVID'),
    }

    res = (640,360)
    fps = 24

    logging.info("Creating VideoWriter to: {}, {}fps, {}x{}px".format(outvideo, fps, res[0], res[1]))
    disparityWriter = cv2.VideoWriter(outvideo, VIDEO_TYPE[os.path.splitext(outvideo)[1][1:]], fps, res)

  computerValues = [
    0, #16, # minDisp
    16, #112-16, # numDisp
    3, #3, # blockSize
    0, #8*3*3**2, # p1
    0, #32*3*3**2, # p2
    0, #1, #disp12MaxDiff
    0, #10, #uniquenessRatio
    0, #100, #speckleWindowSize
    0 #32 #speckleRange
  ]

  computer = Computer(*computerValues)

  # GUI


  blockSizeVals = [1,3,5,7,9,11,13,15,17,19]
  cv2.namedWindow("GUI")

  
  def onMinDisp(val):    
    computerValues[0] = val
    computer.set(*computerValues)

  cv2.createTrackbar("minDisp", "GUI", 16, 100, onMinDisp)


  def onDispVal(val):    
    numDisparities = 16 * (1 + math.floor(float(val) / 100.0 * 15))
    computerValues[1] = numDisparities
    computer.set(*computerValues)

  cv2.createTrackbar("numDisparities", "GUI", 100, 100, onDispVal)


  def onBlockSize(val):    
    computerValues[2] = blockSizeVals[val]
    computer.set(*computerValues)

  cv2.createTrackbar("blockSize", "GUI", 1, len(blockSizeVals)-1, onBlockSize)


  def onDiff(val):    
    computerValues[2] = val-1
    computer.set(*computerValues)

  cv2.createTrackbar("disp12MaxDiff", "GUI", computerValues[5]+1, 101, onDiff)

  def onRatio(val):    
    computerValues[6] = val
    computer.set(*computerValues)

  cv2.createTrackbar("uniquenessRatio", "GUI", computerValues[6], 100, onRatio)


  def onSpeckleWindowSize(val):    
    computerValues[7] = val
    computer.set(*computerValues)

  cv2.createTrackbar("speckleWindowSize", "GUI", computerValues[7], 200, onSpeckleWindowSize)

  def onSpeckleRange(val):    
    computerValues[8] = val
    computer.set(*computerValues)

  cv2.createTrackbar("speckleRange", "GUI", computerValues[8], 6, onSpeckleRange)

  
  logging.info("Starting playback, press <ESC> or 'Q' or CTRL+C to stop, <SPACE> to pause and 'S' to save a frame...")
  isPaused = False
  saveframe = False

  def disparityFrameCallback(frame):
    cv2.imshow("DISPARITY", (frame - computerValues[0]) / computerValues[1])

    # if not saveframe:
      # return
    if disparityWriter:
      disparityWriter.write(frame)

  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if time.time() > nextFrameTime:
          
          isDone = update(streams, computer, crop, showinput, disparityFrameCallback)

          if isDone and loop:
            isDone = False
            for s in streams:
              s.init()
            logging.info('loop')

          saveframe = False
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

      if isDone:
        break

  except KeyboardInterrupt:
    logging.info("KeyboardInterrupt, stopping")

  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = OptionParser()

  parser.add_option("-l", "--input-video-left", dest="leftvideo",
                    help="Video file to read from, (default: %default)",
                    default=DEFAULTS['invideos'][0])

  parser.add_option("-r", "--input-video-right", dest="rightvideo",
                    help="Video file to read from, (default: %default)",
                    default=DEFAULTS['invideos'][1])

  parser.add_option("-o", "--output-video", dest="outvideo", type="string",
                    help="Path to file where undistorted video should be saved",
                    default=None)

  parser.add_option("-d", "--delay", dest="delay", type="float",
                    help="Delay between each frame",
                    default=DEFAULTS['delay'])

  parser.add_option("-c", "--crop",
                    action="store_true", dest="crop", default=DEFAULTS['crop'],
                    help="Crop undistorted images when previewing calibration results")

  parser.add_option("-L", "--Loop",
                    action="store_true", dest="loop", default=DEFAULTS['loop'],
                    help="Loop input videos")

  parser.add_option("-s", "--show-input",
                    action="store_true", dest="showinput", default=False,
                    help="Show input videos")

  parser.add_option("-f", "--calibration-file", dest="calibfile", type="string",
                    help="Path calibration file",
                    default=DEFAULTS['calibfile'])

  parser.add_option("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Verbose logging to stdout")
  

  (options, args) = parser.parse_args()

  main(video_paths=(options.leftvideo, options.rightvideo), calibrationFilePath=options.calibfile, crop=options.crop, delay=options.delay, verbose=options.verbose, outvideo=options.outvideo, loop=options.loop, showinput=options.showinput)




