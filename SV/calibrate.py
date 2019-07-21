# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html


import os, time
import numpy as np
import cv2
from optparse import OptionParser

DEFAULTS = {
  'video': None,
  'delay': None,
  'x_amount': 9, 'y_amount': 6,
  'square_size': 1.0
}

class Context:
  def __init__(self):
    pass



def parse_args():
  parser = OptionParser()
  parser.add_option("-v", "--video", dest="video",
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

  (options, args) = parser.parse_args()
  return (options, args)

def get_LR_filenames(filename):
  filename, ext = os.path.splitext(filename)
  return (filename+'_L'+ext, filename+'_R'+ext)  

def get_stream(vid_path, pattern_dimm, size):
  cap = cv2.VideoCapture(vid_path)
  return {'cap': cap, 'ID': vid_path, 'pattern_dimm': pattern_dimm,
    'allFramesProcessed': False, 'last_found_frame': None, 'frame_corners': [], 'square_size': size, 'calibrateResult': None}

def get_calibration_data(image_points, pattern_dimm, image_dimms, pattern_square_size=1.0):
  # ## https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
  # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
  # objp = np.zeros((pattern_dimm[0]*pattern_dimm[1],3), np.float32)
  # objp[:,:2] = np.mgrid[0:pattern_dimm[0],0:pattern_dimm[1]].T.reshape(-1,2)

  # https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py#L49
  objp = np.zeros((np.prod(pattern_dimm), 3), np.float32)
  objp[:, :2] = np.indices(pattern_dimm).T.reshape(-1, 2)
  objp *= pattern_square_size

  pattern_points = []
  for p in image_points:
    pattern_points.append(objp)

  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(pattern_points,  image_points, image_dimms, None, None)
  result = (ret, mtx, dist, rvecs, tvecs)
  return result

def update(streams):
  for s in streams:
    if not s['calibrateResult'] == None:
      continue

    if s['allFramesProcessed']:
      result = []

      if len(s['frame_corners']) == 0:
        print("Didn't find any chessboards for: {}".format(s['ID']))
      else:
        imgdimms = s['last_found_frame'].shape[::-1]
        print("Calibrating {} using checkerboard points from {} frames and image dimmensions: {}".format(s['ID'], len(s['frame_corners']), imgdimms))
        result = get_calibration_data(s['frame_corners'], s['pattern_dimm'], imgdimms, s['square_size'])
        print("calibrate result for {}:\n{}\n\n".format(s['ID'], result))

      s['calibrateResult'] = result
      continue

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
        s['last_found_frame'] = gray
      else:
        cv2.imshow(s['ID'], frame)

      continue

    s['allFramesProcessed'] = True  

  return len(list(filter(lambda s: s['calibrateResult'] == None, streams))) == 0

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


if __name__ == '__main__':
  (options, args) = parse_args()
  print(options)

  streams = []

  if options.video:
    save_path_L, save_path_R = get_LR_filenames(options.video)
    print('Using videos: {}, {}'.format(save_path_L,save_path_R))
    streams.append(get_stream(save_path_L, (options.x_amount, options.y_amount), options.square_size))
    streams.append(get_stream(save_path_R, (options.x_amount, options.y_amount), options.square_size))

  print("Starting calibration, press 'Q' or CTRL+C to stop...")
  isPaused = False

  try:
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if not options.delay or time.time() > nextFrameTime:
          isDone = update(streams)
          nextFrameTime = time.time() + options.delay

      if isDone:
        break

      key = cv2.waitKey(20) & 0xFF
      if key == ord('q'):
        break

      if key == ord(' '):
        isPaused = not isPaused
      
      if key == ord('c'): # continue
        for s in streams:
          s['allFramesProcessed'] = True

  except KeyboardInterrupt:
    print("KeyboardInterrupt, stopping")
    
  for s in streams:
    s['cap'].release()

  cv2.destroyAllWindows()





