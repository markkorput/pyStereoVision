# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json
import numpy as np
from optparse import OptionParser
from datetime import datetime
from .utils.CalibrationFile import CalibrationFile

DEFAULTS = {
  'video': 'saved-media/base75mm-pattern22mm-short_L.avi',
  'delay': None,
  'calibfile': 'saved-media/calibration.json',
  'crop': False
}

class Stream:
  def __init__(self, vid_path, calibdata, outvideo):
    self.id = vid_path
    self.cap = cv2.VideoCapture(vid_path)
    self.calibrationdata = calibdata
    self.done = False
    self.writer = None

    if outvideo:
      if outvideo == 'auto':
        filename, ext = os.path.splitext(vid_path)
        outvideo = '{}-UNDISTORTED{}'.format(filename,ext)

      VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        #'mp4': cv2.VideoWriter_fourcc(*'H264'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
      }

      res = (640,360)
      fps = 24

      logging.info("Creating VideoWriter to: {}, {}fps, {}x{}px".format(outvideo, fps, res[0], res[1]))
      self.writer = cv2.VideoWriter(outvideo, VIDEO_TYPE[os.path.splitext(outvideo)[1][1:]], fps, res)

  def __del__(self):
    if self.cap:
      self.cap.release()
      self.cap = None
    if self.writer:
      self.writer.release()
      self.writer = None


def get_undistort(img, calibdata, crop=True):
  ret, mtx, dist, rvecs, tvecs = calibdata
  h,  w = img.shape[:2]  
  # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

  # undistort
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  
  if not crop:
    return dst

  # crop the image
  x,y,w,h = roi
  return dst[y:y+h, x:x+w]

def update(streams, crop, frameCallback):
  for s in streams:
    # already have calibration result? show undistorted image
    if not s.calibrationdata or len(s.calibrationdata) == 0:
      s.done = True
    else:
      (retval, frame) = s.cap.read()
      if retval:
        undistorted_image = get_undistort(frame, s.calibrationdata, crop=crop)
        #cv2.imwrite(id,dst)
        cv2.imshow("{} - UNDISTORTED".format(s.id), undistorted_image)
        if s.writer:
          s.writer.write(undistorted_image)
        frameCallback(s.id, frame, undistorted_image)
      else:
        s.done = True

  return len(list(filter(lambda s: s.done == False, streams))) == 0

def main(video_paths, calibrationFilePath=None, crop=True, delay=0, verbose=False, outvideo=None):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  streams = []
  calibfile = CalibrationFile(calibrationFilePath)

  for vid_path in video_paths:
    calibdata = calibfile.getDataForVideoId(vid_path)
    streams.append(Stream(vid_path, calibdata, outvideo))

  logging.info("Starting undistort playback, press <ESC> or 'Q' or CTRL+C to stop, <SPACE> to pause and 'S' to save a frame...")
  isPaused = False
  saveframe = False

  def frameCallback(videoId, frame,undistorted):
    if not saveframe:
      return
    logging.info('saving frames...')
    prefix = '{}-frame-{}'.format(videoId, datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
    p1 = '{}-original.png'.format(prefix)
    p2 = '{}-undistorted.png'.format(prefix)
    cv2.imwrite(p1, frame)
    logging.info('original saved to {}'.format(p1))
    cv2.imwrite(p2, undistorted)
    logging.info('undistorted saved to {}'.format(p2))


  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if time.time() > nextFrameTime:
          
          isDone = update(streams, crop, frameCallback)
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

  parser.add_option("-i", "--input-video", dest="video",
                    help="Video file to read from, (default: %default)",
                    default=DEFAULTS['video'])

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

  main(video_paths=[options.video], calibrationFilePath=options.calibfile, crop=options.crop, delay=options.delay, verbose=options.verbose, outvideo=options.outvideo)




