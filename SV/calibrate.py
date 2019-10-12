# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html


import os, time
import numpy as np
import cv2
from optparse import OptionParser
import json

DEFAULTS = {
  'video': 'saved-media/base75mm-pattern22mm-short.avi',
  'delay': None,
  'x_amount': 9, 'y_amount': 6,
  'square_size': 1.0,
  'calibfile': 'saved-media/calibration.json'
}

class Context:
  def __init__(self):
    pass


def get_LR_filenames(filename):
  filename, ext = os.path.splitext(filename)
  return (filename+'_L'+ext, filename+'_R'+ext)  

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

def getCalibData(filepath, videoId):
  if not os.path.isfile(filepath):
    return None

  text = None
  with open(filepath, "r") as f:
      text = f.read()

  data = None
  try:
    data = json.loads(text)
  except json.decoder.JSONDecodeError as err:
    print('Could not load calibration json: \n{}'.format(err))
    return None

  if not 'sv' in data:
    return None
  if not 'calibration_data' in data['sv']:
    return None
  if not videoId in data['sv']['calibration_data']:
    return None
  serialized = data['sv']['calibration_data'][videoId]
  return fromSerializable(serialized)


def fromSerializable(data):
  return [
    data[0],
    np.array(data[1]),
    np.array(data[2]),
    list(map(lambda i: np.array(i), data[3])),
    list(map(lambda i: np.array(i), data[4]))
  ]

def toSerializable(data):
  return [
    data[0],
    data[1].tolist(),
    data[2].tolist(),
    list(map(lambda i: i.tolist(), data[3])),
    list(map(lambda i: i.tolist(), data[4]))
  ]

def saveCalibData(filepath, data, videoId):
  text = '{}'
  if os.path.isfile(filepath):
    with open(filepath, "r") as f:
      text = f.read()
  
  json_data = {}
  try:
    json_data = json.loads(text)
  except json.decoder.JSONDecodeError as err:
    # print('Could not load calibration json: \n{}'.format(err))
    json_data = {}

  # json.set(['sv','calibration_data',videoId], json_data)
  if not 'sv' in json_data:
    json_data['sv'] = {}
  if not 'calibration_data' in json_data['sv']:
    json_data['sv']['calibration_data'] = {}

  json_data['sv']['calibration_data'][videoId] = toSerializable(data)

  with open(filepath, "w") as f:
    # f.write(json.dumps(json_data))
    json.dump(json_data, f)

  print('Wrote calibration data for {} to: {}'.format(videoId, filepath))

def update(streams, crop=False, calibFile=None):
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
          s['last_found_frame'] = gray
        else:
          cv2.imshow(s['ID'], frame)

      else:
        s['allFramesProcessed'] = True  
        print("All frames processed for {}".format(s['ID']))
        cv2.destroyAllWindows()

      continue

    # do we have the chessboard data for all frames? then generate calibration data
    # from those frame chessboards
    if s['calibrateResult'] == None:
      result = []

      if len(s['frame_corners']) == 0:
        print("Didn't find any chessboards for: {}".format(s['ID']))
      else:
        imgdimms = s['last_found_frame'].shape[::-1]
        print("Calibrating camera for {} using checkerboard points from {} frames and image dimmensions: {}".format(s['ID'], len(s['frame_corners']), imgdimms))
        result = get_calibration_data(s['frame_corners'], s['pattern_dimm'], imgdimms, s['square_size'])
        print("calibrate result for {}:\n{}\n\n".format(s['ID'], result))
        if calibFile:
          saveCalibData(calibFile, result, s['ID'])

      s['calibrateResult'] = result
      continue

    # already have calibration result? show undistorted image
    if len(s['calibrateResult']) == 0:
      s['done'] = True
    else:
      (retval, frame) = s['cap'].read()
      if retval:
        undistorted_image = get_undistort(frame, s['calibrateResult'], crop=crop)
        #cv2.imwrite(id,dst)
        cv2.imshow("{} - UNDISTORTED".format(s['ID']), undistorted_image)
      else:
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


def get_stream(vid_path, pattern_dimm, size, calibrationFilePath=None):
  cap = cv2.VideoCapture(vid_path)
  calibres = getCalibData(calibrationFilePath, vid_path) if calibrationFilePath else None
  if calibres:
    print("Found calibration results for {} in {}".format(vid_path, calibrationFilePath))
  return {'cap': cap, 'ID': vid_path, 'pattern_dimm': pattern_dimm,
    'allFramesProcessed': False, 'last_found_frame': None, 'frame_corners': [], 'square_size': size, 'calibrateResult': calibres, 'done': False}

def main(video_paths, grid_size, square_size, calibrationFilePath=None, crop=True, delay=0):
  streams = []

  for p in video_paths:
    streams.append(get_stream(p, (grid_size[0], grid_size[1]), square_size, calibrationFilePath))

  print("Starting calibration, press 'Q' or CTRL+C to stop...")
  isPaused = False

  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    while(True):
      if not isPaused:
        if time.time() > nextFrameTime:
          isDone = update(streams, crop=crop, calibFile=calibrationFilePath)
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
    print("KeyboardInterrupt, stopping")
    
  for s in streams:
    s['cap'].release()

  cv2.destroyAllWindows()


if __name__ == '__main__':
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

  parser.add_option("-c", "--crop",
                    action="store_true", dest="crop", default=False,
                    help="Crop undistorted images when previewing calibration results")

  parser.add_option("-f", "--calibration-file", dest="calibfile", type="string",
                    help="Path calibration file",
                    default=DEFAULTS['calibfile'])


  (options, args) = parser.parse_args()

  main(video_paths=get_LR_filenames(options.video), calibrationFilePath=options.calibfile, grid_size=(options.x_amount, options.y_amount), square_size=options.square_size, crop=options.crop, delay=options.delay)




