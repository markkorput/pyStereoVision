# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html


import os, time
# import numpy as np
import cv2
from optparse import OptionParser

DEFAULTS = {
  'video': None,
  'delay': None,
  'x_amount': 9, 'y_amount': 6
}

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

  (options, args) = parser.parse_args()
  return (options, args)

def get_LR_filenames(filename):
  filename, ext = os.path.splitext(filename)
  return (filename+'_L'+ext, filename+'_R'+ext)  

def get_stream(vid_path, pattern_dimm):
  cap = cv2.VideoCapture(vid_path)
  return {'cap': cap, 'ID': vid_path, 'pattern_dimm': pattern_dimm}

def update(streams):
  for s in streams:
    (retval, frame) = s['cap'].read()
    if not retval:
      continue

    chess_ret, corners = get_corners(frame, s['pattern_dimm'])

    if chess_ret:
      overlay_img = cv2.drawChessboardCorners(frame, s['pattern_dimm'], corners, chess_ret)
      cv2.imshow(s['ID'], overlay_img)
    else:
      cv2.imshow(s['ID'], frame)

def get_corners(img, pattern_dimm):
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  # Find the chess board corners
  ret, corners = cv2.findChessboardCorners(gray, pattern_dimm, None)

  # termination criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

  # If found, add object points, image points (after refining them)
  if ret == True:
    return ret, cv2.cornerSubPix(gray,corners,pattern_dimm,(-1,-1),criteria)
  return ret, corners


if __name__ == '__main__':
  (options, args) = parse_args()
  print(options)

  streams = []

  if options.video:
    save_path_L, save_path_R = get_LR_filenames(options.video)
    print('Using videos: {}, {}'.format(save_path_L,save_path_R))
    streams.append(get_stream(save_path_L, (options.x_amount, options.y_amount)))
    streams.append(get_stream(save_path_R, (options.x_amount, options.y_amount)))

  print("Starting calibration, press 'Q' or CTRL+C to stop...")
  isPaused = False

  try:
    nextFrameTime = time.time()
    while(True):
      if not isPaused:
        if not options.delay or time.time() > nextFrameTime:
          update(streams)
          nextFrameTime = time.time() + options.delay

      key = cv2.waitKey(20) & 0xFF
      if key == ord('q'):
        break

      if key == ord(' '):
        isPaused = not isPaused

  except KeyboardInterrupt:
    print("KeyboardInterrupt, stopping")
    
  for s in streams:
    s['cap'].release()

  cv2.destroyAllWindows()





