# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json, threading
import numpy as np
from optparse import OptionParser
from .utils.CalibrationFile import CalibrationFile

DEFAULTS = {
  'video': 'saved-media/base75mm-pattern22mm-short_L.avi',
  'delay': None,
  'x_amount': 9, 'y_amount': 6,
  'square_size': 0.022,
  'calibfile': 'saved-media/calibration.json'
}

def get_calibration_data(image_points, pattern_dimm, image_dimms, pattern_square_size=1.0):
  # ## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  # objp = np.zeros((pattern_dimm[0]*pattern_dimm[1],3), np.float32)
  # objp[:,:2] = np.mgrid[0:pattern_dimm[0],0:pattern_dimm[1]].T.reshape(-1,2)

  # https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py#L49
  objp = np.zeros((np.prod(pattern_dimm), 3), np.float32)
  objp[:, :2] = np.indices(pattern_dimm).T.reshape(-1, 2)
  objp *= pattern_square_size

  logging.debug("Generated chessboard object points for calibrateCamera:\n{}".format(objp))
  pattern_points = []
  for p in image_points:
    pattern_points.append(objp)

  logging.info("Running cv2.calibrateCamera")
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(pattern_points,  image_points, image_dimms, None, None)
  logging.info("cv2.calibrateCamera finished, ret val: {}".format(ret))
  #result = (ret, mtx, dist, rvecs, tvecs)
  return ret, mtx, dist, rvecs, tvecs #result

def update(streams, crop=False, calibfile=None):
  for s in streams:
    # process each frame; try to find a chessboard pattern and save the data if found
    if s['calibrateResult'] == None and not s['allFramesProcessed']:
      (retval, frame) = s['cap'].read()
      if retval:
        found, corners, subcorners, gray = get_corners(frame, s['pattern_dimm'])
        if found:
          # show found corners to user
          overlay_img = cv2.drawChessboardCorners(frame, s['pattern_dimm'], subcorners, found)
          cv2.imshow(s['ID'], overlay_img)

          # save found corners
          ## see: https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py#L83
          s['frame_corners'].append(corners.reshape(-1,2))
          print("Found {} of frames with chessboard patterns for {}\r".format(len(s['frame_corners']), s['ID']), end="")
          s['last_found_frame'] = gray
        else:
          cv2.imshow(s['ID'], frame)

      else:
        s['allFramesProcessed'] = True  
        logging.info("All frames processed for {}".format(s['ID']))
        cv2.destroyAllWindows()

      continue

    # do we have the chessboard data for all frames? then generate calibration data
    # from those frame chessboards
    if s['calibrateResult'] == None and not s['calibrationThread']:
      result = []

      if len(s['frame_corners']) == 0:
        logging.info("Didn't find any chessboards for: {}".format(s['ID']))
      else:
        def threadFunc():
          imgdimms = s['last_found_frame'].shape[::-1]
          logging.info("Calibrating camera for {} using checkerboard points from {} frames and image resolution {}x{}".format(s['ID'], len(s['frame_corners']), imgdimms[0], imgdimms[1]))
          ret, mtx, dist, rvecs, tvecs = get_calibration_data(s['frame_corners'], s['pattern_dimm'], imgdimms, s['square_size'])
          s['calibrateResult'] = (ret, mtx, dist, rvecs, tvecs)
          logging.debug("calibrate result for {}:\n{}\n\n".format(s['ID'], s['calibrateResult']))
          if calibfile:
            calibfile.setDataForVideoId(s['ID'], s['calibrateResult'])
            logging.info("calibration data for {} saved to {}".format(s['ID'], calibfile.filepath))

      thread = threading.Thread(target=threadFunc)
      thread.start()
      s['calibrationThread'] = thread
      s['starttime'] = time.time()
      continue

    if s['calibrationThread']:
      if s['calibrationThread'].isAlive():
        print("Calibrating... ({0:.2f}s)\r".format(time.time() - s['starttime']), end='')
      else:
        s['calibrationThread'] = None
        print('Calibration done.')
        
    if s['calibrateResult'] and not s['calibrationThread']:
      s['done'] = True

  return len(list(filter(lambda s: s['done'] == False, streams))) == 0

def get_corners(img, pattern_dimm):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Find the chess board corners
  found, corners = cv2.findChessboardCorners(gray, pattern_dimm, None)

  # If found, add object points, image points (after refining them)
  if not found:
    return found, corners, None, gray

  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
  subcorners = cv2.cornerSubPix(gray,corners,pattern_dimm,(-1,-1),criteria)
  return found, corners, subcorners, gray

def main(video_paths, grid_size, square_size, calibrationFilePath=None, crop=True, delay=0, verbose=False):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  calibfile = CalibrationFile(calibrationFilePath)
  streams = []

  for vid_path in video_paths:
    cap = cv2.VideoCapture(vid_path)
    calibres = calibfile.getDataForVideoId(vid_path)
    if calibres:
      logging.info("Found calibration results for {} in {}".format(vid_path, calibrationFilePath))
    stream = {'cap': cap, 'ID': vid_path, 'pattern_dimm': (grid_size[0], grid_size[1]),
      'allFramesProcessed': False, 'last_found_frame': None, 'frame_corners': [],
      'square_size': square_size, 'calibrateResult': calibres,
      'calibrationThread': None, 'done': False, 'starttime': None}
    streams.append(stream)

  logging.info("Starting calibration, press 'Q' or CTRL+C to stop...")
  isPaused = False

  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if time.time() > nextFrameTime:
          isDone = update(streams, crop=crop, calibfile=calibfile)
          if isDone:
            logging.info("Finished")
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

  parser.add_option("-x", "--x-amount", dest="x_amount", type="int",
                    help="Number of columns in the chessboard pattern",
                    default=DEFAULTS['x_amount'])

  parser.add_option("-y", "--y-amount", dest="y_amount", type="int",
                    help="Number of rows in the chessboard pattern",
                    default=DEFAULTS['y_amount'])

  parser.add_option("-s", "--square-size", dest="square_size", type="float",
                    help="Size of each square",
                    default=DEFAULTS['square_size'])

  parser.add_option("-c", "--crop",
                    action="store_true", dest="crop", default=False,
                    help="Crop undistorted images when previewing calibration results")

  parser.add_option("-f", "--calibration-file", dest="calibfile", type="string",
                    help="Path calibration file",
                    default=DEFAULTS['calibfile'])

  parser.add_option("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Verbose logging to stdout")

  (options, args) = parser.parse_args()

  main(video_paths=[options.video], calibrationFilePath=options.calibfile, grid_size=(options.x_amount, options.y_amount), square_size=options.square_size, crop=options.crop, delay=options.delay, verbose=options.verbose)




