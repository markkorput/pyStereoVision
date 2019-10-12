# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json
import numpy as np
from optparse import OptionParser
from datetime import datetime
from .utils.CalibrationFile import CalibrationFile

DEFAULTS = {
  'invideos': ['saved-media/base75mm-pattern22mm-short_L-UNDISTORTED.avi','saved-media/base75mm-pattern22mm-short_R-UNDISTORTED.avi'],
  'delay': None,
  'calibfile': None, #'saved-media/calibration.json',
  'crop': False
}

class Stream:
  def __init__(self, vid_path, calibdata):
    self.id = vid_path
    self.cap = cv2.VideoCapture(vid_path)
    self.calibrationdata = calibdata
    self.done = False

  def __del__(self):
    if self.cap:
      self.cap.release()
      self.cap = None


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

def update(streams, computer, crop, disparityFrameCallback):
  frames = []

  # for s in streams:
  for s in streams:
    (retval, frame) = s.cap.read()
    if retval:
      undistorted_image = get_undistort(frame, s.calibrationdata, crop=crop) if s.calibrationdata else frame
      cv2.imshow("{} - UNDISTORTED".format(s.id), undistorted_image)
      #if s.writer:
      #  s.writer.write(undistorted_image)
      frames.append(undistorted_image)
    else:
      s.done = True

  if len(frames) == 2:
    logging.info('Computing disparity...')
    disp = computer.compute(frames[0], frames[1]).astype(np.float32) / 16.0

    # disp = getDisparity(pair[0], pair[1])
    cv2.imshow("DISPARITY", disp)
    # cv2.imshow('DISPARITY', (disp-min_disp)/num_disp)
    disparityFrameCallback(disp)


  return len(list(filter(lambda s: s.done == True, streams))) > 0

def main(video_paths, calibrationFilePath=None, crop=True, delay=0, verbose=False, outvideo=None):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  streams = []
  calibfile = CalibrationFile(calibrationFilePath) if calibrationFilePath else None

  for vid_path in video_paths:
    calibdata = calibfile.getDataForVideoId(vid_path) if calibfile else None
    streams.append(Stream(vid_path, calibdata))


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

  computer = cv2.StereoSGBM_create()

  logging.info("Starting playback, press <ESC> or 'Q' or CTRL+C to stop, <SPACE> to pause and 'S' to save a frame...")
  isPaused = False
  saveframe = False

  def disparityFrameCallback(frame):
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
          
          isDone = update(streams, computer, crop, disparityFrameCallback)
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

  parser.add_option("-f", "--calibration-file", dest="calibfile", type="string",
                    help="Path calibration file",
                    default=DEFAULTS['calibfile'])

  parser.add_option("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Verbose logging to stdout")
  

  (options, args) = parser.parse_args()

  main(video_paths=(options.leftvideo, options.rightvideo), calibrationFilePath=options.calibfile, crop=options.crop, delay=options.delay, verbose=options.verbose, outvideo=options.outvideo)




