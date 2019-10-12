# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json
import numpy as np
from optparse import OptionParser
from .utils.CalibrationFile import CalibrationFile

DEFAULTS = {
  'video': 'saved-media/base75mm-pattern22mm-short_L.avi',
  'delay': None,
  'calibfile': 'saved-media/calibration.json',
  'crop': False
}

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

def update(streams, crop=False):
  for s in streams:
    # already have calibration result? show undistorted image
    if len(s['calibrationData']) == 0:
      s['done'] = True
    else:
      (retval, frame) = s['cap'].read()
      if retval:
        undistorted_image = get_undistort(frame, s['calibrationData'], crop=crop)
        #cv2.imwrite(id,dst)
        cv2.imshow("{} - UNDISTORTED".format(s['ID']), undistorted_image)
      else:
        s['done'] = True

  return len(list(filter(lambda s: s['done'] == False, streams))) == 0

def main(video_paths, calibrationFilePath=None, crop=True, delay=0, verbose=False):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  streams = []
  calibfile = CalibrationFile(calibrationFilePath)

  for vid_path in video_paths:
    cap = cv2.VideoCapture(vid_path)
    calibdata = calibfile.getDataForVideoId(vid_path)
    stream = {'cap': cap, 'ID': vid_path, 'calibrationData': calibdata, 'done': False}
    streams.append(stream)

  logging.info("Starting undistort playback, press 'Q' or CTRL+C to stop...")
  isPaused = False

  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if time.time() > nextFrameTime:
          isDone = update(streams, crop=crop)
          if delay:
            nextFrameTime = time.time() + delay

      # process user input
      key = cv2.waitKey(20) & 0xFF
      if key == ord('q'):
        isDone = True

      if key == ord(' '):
        isPaused = not isPaused

      if key == ord('c'): # continue
        for s in streams:
          s['allFramesProcessed'] = True

      if isDone:
        break

  except KeyboardInterrupt:
    logging.info("KeyboardInterrupt, stopping")
    
  for s in streams:
    s['cap'].release()

  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = OptionParser()

  parser.add_option("-i", "--video", dest="video",
                    help="Video file to read from, (default: %default)",
                    default=DEFAULTS['video'])

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

  main(video_paths=[options.video], calibrationFilePath=options.calibfile, crop=options.crop, delay=options.delay, verbose=options.verbose)




