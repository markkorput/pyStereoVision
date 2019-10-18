# Based on:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
import os, time, logging, cv2, json, math
import numpy as np
from optparse import OptionParser
from datetime import datetime
from .utils import isNone
from .utils.CalibrationFile import CalibrationFile
from .undistort import get_undistort

class Stream:
  def __init__(self, vid_path, calibdata, gray=True):
    self.id = vid_path
    self.calibrationdata = calibdata
    self.init()
    self.lastFrame = None

  def init(self):
    self.cap = cv2.VideoCapture(int(self.id) if self.id.isdigit() else self.id)
    self.done = False

  def __del__(self):
    if self.cap:
      self.cap.release()
      self.cap = None

class Computer(dict):
  def __init__(self, params=None):
    self.params = self
    self.update({
      'name': 'Default',
      'enabled': True,
      'minDisp': 0,
      'numDisp': 16,
      'blockSize': 3,
      'windowSize': 0,
      # 'p1': 0,
      # 'p2': 0,
      'disp12MaxDiff': 0,
      'uniquenessRatio': 0,
      'speckleWindowSize': 0,
      'preFilterCap': 63,
      'speckleRange': 32,

      'wls-enabled': False,
      'wls-normalize': True,
      'wls-show-confmap': False
    })

    if params:
      self.params.update(params)
    
    self.isDirty = True

  def __setitem__(self, key, value):
    super().__setitem__(key, value)
    self.isDirty = True
    self.stereo = None
    self.leftMatcher = None
    self.rightMatcher = None
    self.wls_filter = None

  def compute(self, imgL, imgR):
    # if self.timotheus:
    #   return self.timotheusCompute(*args)
    dirty = self.isDirty
    self.isDirty=False

    if dirty:
      logging.info('(re-)initializing stereo')
      self.stereo = cv2.StereoSGBM_create(
        minDisparity=self.params['minDisp'],
        numDisparities=self.params['numDisp'],
        blockSize=self.params['blockSize'],
        P1=8 * 3 * self.params['windowSize'] ** 2, # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * self.params['windowSize'] ** 2,
        disp12MaxDiff=self.params['disp12MaxDiff'],
        uniquenessRatio = self.params['uniquenessRatio'],
        speckleWindowSize = self.params['speckleWindowSize'],
        speckleRange = self.params['speckleRange'],
        preFilterCap=self.params['preFilterCap'],
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    if not self.params['wls-enabled']:
      return self.stereo.compute(imgL, imgR)

    # http://timosam.com/python_opencv_depthimage

    if dirty:
      self.leftMatcher = self.stereo
      self.rightMatcher = cv2.ximgproc.createRightMatcher(self.leftMatcher)

      # FILTER Parameters
      lmbda = 80000
      sigma = 1.2
      visual_multiplier = 1.0
      
      self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.leftMatcher)
      self.wls_filter.setLambda(lmbda)
      self.wls_filter.setSigmaColor(sigma)
      # Now we can compute the disparities and conve
      
    displ = self.leftMatcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = self.rightMatcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = self.wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    # Finally if you show this image with imshow() you may not see anything. This is due to values being not normalized to a 8-bit format. So lets fix this by normalizing our depth map:

    if self.params['wls-normalize']:
      filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);

    if self.params['wls-show-confmap']:
      conf_map = self.wls_filter.getConfidenceMap()
      cv2.imshow('confmap [{}]'.format(self.params['name']), conf_map)

    # cv2.imshow('Disparity Map', filteredImg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return filteredImg

  def getStereoDisparity(self, frames):
    '''
    Takes a list of two frames (left and right respectively) and returns a frame with the disparity (grayscale depth image) of the two frames

    Args:
      frames (list): two input frames, the first one should be the left "eye", the second one should be the right "eye"
    '''

    # make sure both frames have the same size
    h1,w1 = frames[0].shape[:2]
    h2,w2 = frames[1].shape[:2]

    if h1 != h2 or w1 != w2:
      logging.info('Resizing left frame...') 
      frames[0] = cv2.resize(frames[0], (w2,h2))

    logging.debug('Computing disparity...')
    disp = self.compute(frames[0], frames[1]) #.astype(np.float32) / 16.0

    # disp = getDisparity(pair[0], pair[1])
    # cv2.imshow("DISPARITY", disp)
    # cv2.imshow('DISPARITY', (disp-min_disp)/num_disp)
    # disparityFrameCallback(disp)
    return disp

def update(streams, computers, params, crop, disparityFrameCallback):

  # fetch image for all (both?) streams
  for s in streams:
    (retval, frame) = s.cap.read()
    if retval:
      s.lastFrame = frame
    else:
      s.done = True

  frames = []

  for s in streams:
    f = s.lastFrame
    if type(f) == type(None): continue

    #for idx, f in enumerate(frames):
    if params['gray']:
      logging.debug("Converting to grayscale...")
      f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    if params['resize-enabled']:
      f = cv2.resize(f, (params['resize-width'],params['resize-height']))

    if s.calibrationdata:
      logging.debug("Applying calbiration...")
      f = get_undistort(f, s.calibrationdata, crop=crop)

    if params['show-input']:
      cv2.imshow("Input: {}".format(s.id), f)

    frames.append(f)

  if len(frames) == 2:
    for computer in computers:
      if not computer.params['enabled']:
        continue
      disp = computer.getStereoDisparity(frames)

      # disp = getDisparity(pair[0], pair[1])
      # cv2.imshow("DISPARITY", disp)
      # cv2.imshow('DISPARITY', (disp-min_disp)/num_disp)
      disparityFrameCallback(disp, computer)

  return len(list(filter(lambda s: s.done == True, streams))) > 0

from SV.utils import addParamTrackbar, createParamsGuiWin

def createGui(params, computers):
  winid = 'App'

  cv2.namedWindow(winid)
  cv2.moveWindow(winid, 5, 5)
  cv2.resizeWindow(winid, 500,400)

  
  with createParamsGuiWin('App', params) as gui:
    gui.add('delay', factor=1000)
    gui.add('gray', values=[False,True])
    gui.add('calibrate-enabled', values=[False,True])
    
    gui.add('resize-enabled', values=[False,True])
    gui.add('resize-width', 1920)
    gui.add('resize-height', 1080)
    
    gui.add('show-input', values=[False,True])
    gui.add('show-disparity', values=[False,True])
    gui.add('write-output', values=[False,True])

  with createParamsGuiWin('Post', params) as gui:
    gui.add('threshold-enabled', values=[False,True])
    gui.add('threshold-thresh', 255)
    gui.add('threshold-maxval', 255)
    gui.add('threshold-type', values=[cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV,cv2.THRESH_MASK])

    gui.add('blur-enabled', values=[False,True])
    gui.add('blur-x', 10)
    gui.add('blur-y', 10)

    gui.add('close-enabled', values=[False,True])
    gui.add('close-x', max=5000)
    gui.add('close-y', max=5000)

    gui.add('dilation-enabled', values=[False,True])
    gui.add('dilation-kernel-value', 20)
    gui.add('dilation-iterations', 10)

    gui.add('contours-enabled', 1)
    gui.add('contours-mode', values=[cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE, cv2.RETR_FLOODFILL])
    gui.add('contours-method', values=[cv2.CHAIN_APPROX_NONE,cv2.CHAIN_APPROX_SIMPLE,cv2.CHAIN_APPROX_TC89_L1,cv2.CHAIN_APPROX_TC89_KCOS])
    gui.add('contours-drawlines', 1)
    gui.add('contours-linethickness', values=[-1,0,1,2,3,4,5])
    gui.add('contours-drawboxes', 1)
    gui.add('contours-minsize', 4000)

    gui.add('canny-enabled', values=[False,True])
    gui.add('canny-threshold1', 500)
    gui.add('canny-threshold2', 500)

    gui.add('lerp-enabled', values=[False,True])
    gui.add('lerp-factor', factor=2000)


  for idx, c in enumerate(computers):
    pars = c.params
    winid = pars['name']

    cv2.namedWindow(winid)
    cv2.moveWindow(winid, 5 + (idx+1) * 10, 5 + (idx+1) * 200)
    cv2.resizeWindow(winid, 500,400)

    addParamTrackbar(winid, pars, 'enabled', values=[False,True])

    addParamTrackbar(winid, pars, 'minDisp', max=100)
    addParamTrackbar(winid, pars, 'numDisp', values=list(map(lambda v: 16*v, range(100))))
    addParamTrackbar(winid, pars, 'blockSize', values=[1,3,5,7,9,11,13,15,17,19,21,23,25,27])
    addParamTrackbar(winid, pars, 'windowSize', max=50)
    addParamTrackbar(winid, pars, 'disp12MaxDiff', values=list(map(lambda v: v-1, range(22))))
    addParamTrackbar(winid, pars, 'uniquenessRatio', max=100)
    addParamTrackbar(winid, pars, 'speckleWindowSize', max=200)
    addParamTrackbar(winid, pars, 'preFilterCap', max=100)
    addParamTrackbar(winid, pars, 'speckleRange', max=6)
    addParamTrackbar(winid, pars, 'wls-enabled', values=[False,True])
    addParamTrackbar(winid, pars, 'wls-normalize', values=[False,True])
    addParamTrackbar(winid, pars, 'wls-show-confmap', values=[False,True])

def main(video_paths, calibrationFilePath=None, crop=True, delay=0, verbose=False, outvideo=None, loop=False, showinput=False, gray=True, resize=None):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  # input streams
  streams = []
  calibfile = CalibrationFile(calibrationFilePath) if calibrationFilePath else None

  for vid_path in video_paths:
    calibdata = calibfile.getDataForVideoId(vid_path) if calibfile else None
    streams.append(Stream(vid_path, calibdata))

  # disparity output writer
  disparityWriter = None

  params = {
    'delay': delay if delay else 0.0,
    'gray': gray,
    'resize-enabled': resize != None,
    'resize-width': 640 if not resize else int((resize if type(resize) in [type([]), type(())] else list(map(lambda v: v.strip(), str(resize).split(','))))[0]),
    'resize-height': 480 if not resize else int((resize if type(resize) in [type([]), type(())] else list(map(lambda v: v.strip(), str(resize).split(','))))[1]),
    'calibrate-enabled': calibfile != None,
    'show-input': showinput,
    'show-disparity': True,
    'write-output': False, # should be "started" manually disparityWriter != None,
    'outWriter': None,

    # post process
    'threshold-enabled': True,
    'threshold-thresh': 80,
    'threshold-maxval': 98,
    'threshold-type': cv2.THRESH_TOZERO, # THRESH_TRUNC, #cv2.THRESH_TRUNC,

    'blur-enabled': False,
    'blur-x': 5,
    'blur-y': 5,

    'dilation-enabled': False,
    'dilation-kernel-value': 6,
    'dilation-iterations': 2,

    'close-enabled': False,
    'close-x': 10,
    'close-y': 10,

    'contours-enabled': False,
    'contours-mode': cv2.RETR_EXTERNAL,
    'contours-method': cv2.CHAIN_APPROX_SIMPLE,
    'contours-drawlines': 0,
    'contours-linethickness': 1,
    'contours-drawboxes': 1,
    'contours-minsize': 0,

    'canny-enabled': False,
    'canny-threshold1': 79,
    'canny-threshold2': 143,

    'lastFrame': None,
    'lerp-enabled': False,
    'lerp-factor': 0.0075
  }

  computers = [
    Computer({
      'enabled': True,
      'wls-enabled': True
    }),
    Computer({
      'name': 'Timotheus',
      'enabled': False,
      'minDisp': 0,
      'numDisp': 160,
      'blockSize': 5,
      'windowSize': 15,
      # 'p1': 0,
      # 'p2': 0,
      'disp12MaxDiff': 1,
      'uniquenessRatio': 15,
      'speckleWindowSize': 0,
      'speckleRange': 2,
      'preFilterCap': 63,
      'wls-enabled': True,
      'wls-normalize': True
    })
  ]

  # GUI
  createGui(params, computers)

  def disparityFrameCallback(frame, computer):

    # threshold?
    if params['threshold-enabled']:
      ret,frame = cv2.threshold(frame, params['threshold-thresh'], params['threshold-maxval'], params['threshold-type'])

    #frame = cv2.getDisparityVis(frame)
    frame = np.uint8(frame)

    # blur?
    if params['blur-enabled'] and params['blur-x'] > 0 and params['blur-y'] > 0:
      frame = cv2.blur(frame, (params['blur-x'],params['blur-y']))

    if params['close-enabled'] and params['close-x'] > 0 and params['close-y'] > 0:
      frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, (params['close-x'],params['close-y']))

    if params['dilation-enabled']:
      val = params['dilation-kernel-value']
      kernel = np.ones((val, val),np.uint8)
      frame = cv2.dilate(frame, kernel, iterations=params['dilation-iterations'])

    if params['contours-enabled']:
      contours,hierarchy = cv2.findContours(frame, params['contours-mode'], params['contours-method'])
      if params['contours-drawlines']:
        frame = cv2.drawContours(frame,contours,-1,(255,255,0),params['contours-linethickness'])
      if params['contours-drawboxes']:
        minsize = params['contours-minsize']
        for c in contours:
          bx,by,bw,bh = cv2.boundingRect(c)
          if minsize == 0 or (bw*bh) >= minsize:
            frame = cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(255,255,0),params['contours-linethickness'])

    # canny edge-detection?
    if params['canny-enabled']:
      frame = cv2.Canny(frame, params['canny-threshold1'], params['canny-threshold2'])

    # lerp?
    if not params['lerp-enabled'] or isNone(params['lastFrame']):
      params['lastFrame'] = frame
    else:
      f = params['lerp-factor']
      params['lastFrame'] = cv2.addWeighted(params['lastFrame'], 1.0 - f, frame, f, 0.0)

    if params['show-disparity']:
      cv2.imshow('DISPARITY [{}]'.format(computer.params['name']), params['lastFrame']) #(frame - computerValues[0]) / computerValues[1])

    if params['write-output']:
      if not params['outWriter'] and outvideo:
        VIDEO_TYPE = {
          'avi': cv2.VideoWriter_fourcc(*'XVID'),
          #'mp4': cv2.VideoWriter_fourcc(*'H264'),
          'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        res = params['lastFrame'].shape[:2]
        fps = 24
        logging.info("Creating VideoWriter to: {}, {}fps, res: {}".format(outvideo, fps, res))
        params['outWriter'] = cv2.VideoWriter(outvideo, VIDEO_TYPE[os.path.splitext(outvideo)[1][1:]], fps, (res[1],res[0]), False)

      if params['outWriter']:
        # outf = cv2.cvtColor(params['lastFrame'], cv2.COLOR_BGR2GRAY)
        params['outWriter'].write(params['lastFrame'])

    elif params['outWriter']:
      params['outWriter'].release()
      params['outWriter'] = None
      logging.info("Close VideoWriter to: {}".format(outvideo))

  try:
    # timing
    nextFrameTime = time.time()
    isDone = False
    isPaused = False
    saveframe = False
    stepFrames = 0
    
    logging.info("Starting playback, press <ESC> or 'Q' or CTRL+C to stop, <SPACE> to pause and 'S' to save a frame...")

    while(True):
      if not isPaused or stepFrames > 0:
        stepFrames = max(0, stepFrames - 1)

        if time.time() > nextFrameTime:

          isDone = update(streams, computers, params, crop, disparityFrameCallback)

          if isDone and loop:
            isDone = False
            for s in streams:
              s.init()
            logging.info('loop')

          saveframe = False
          if params['delay']:
            nextFrameTime = time.time() + params['delay']

      # process user input
      key = cv2.waitKey(20) & 0xFF
      if key != -1 and key != 255:
        if key == 27 or key == ord('q'): # escape or Q
          isDone = True

        elif key == ord(' '):
          isPaused = not isPaused

        elif key == ord('s'):
          saveframe = True

        elif key == ord('n'):
          stepFrames += 1

        elif key == ord('v'):
          verbose = not verbose
          logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')
 
        elif key == ord('r'):
          params['write-output'] = not params['write-output']

        else:
          logging.info('No action for key: {}'.format(key))

      if isDone:
        break

  except KeyboardInterrupt:
    logging.info("KeyboardInterrupt, stopping")

  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = OptionParser()

  parser.add_option("-l", "--input-video-left", dest="leftvideo",
                    help="Video file to read from, (default: %default)",
                    default='saved-media/base75mm-pattern22mm-short_L-UNDISTORTED.avi')

  parser.add_option("-r", "--input-video-right", dest="rightvideo",
                    help="Video file to read from, (default: %default)",
                    default='saved-media/base75mm-pattern22mm-short_R-UNDISTORTED.avi')

  parser.add_option("-o", "--output-video", dest="outvideo", type="string",
                    help="Path to file where undistorted video should be saved",
                    default=None)

  parser.add_option("-d", "--delay", dest="delay", type="float",
                    help="Delay between each frame",
                    default=None)

  parser.add_option("-c", "--crop",
                    action="store_true", dest="crop", default=False,
                    help="Crop undistorted images when previewing calibration results")

  parser.add_option("-L", "--Loop",
                    action="store_false", dest="loop", default=True,
                    help="Don't loop input videos")

  parser.add_option("-s", "--show-input",
                    action="store_true", dest="showinput", default=False,
                    help="Show input videos")

  parser.add_option("-f", "--calibration-file", dest="calibfile", type="string",
                    help="Path calibration file",
                    default=None) #'saved-media/calibration.json',

  parser.add_option("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Verbose logging to stdout")

  parser.add_option("-G", "--no-gray",
                    action="store_false", dest="gray", default=True,
                    help="Convert to grayscale before calculating disparity")
  
  parser.add_option("-n", "--resolution", dest="resize",
                    help="Resize input videos to this resolution, (default: %default)",
                    default=None)


  (options, args) = parser.parse_args()
  main(video_paths=(options.leftvideo, options.rightvideo), calibrationFilePath=options.calibfile, crop=options.crop, delay=options.delay, verbose=options.verbose, outvideo=options.outvideo, loop=options.loop, showinput=options.showinput, gray=options.gray, resize=options.resize)




