from optparse import OptionParser
import cv2, logging
from .utils.processor import create_controlled_processor_from_json_file, create_processor_from_json_file
import numpy as np

logger = logging.getLogger(__name__)


def render(cumm, frame, blend_processor):
  src = frame
  if cumm.shape != frame.shape:
    logging.debug('Resizing to: {}'.format(cumm.shape[:-1]))
    src = cv2.resize(src, (cumm.shape[1], cumm.shape[0]))

  # result = cumm + src
  # result = blend_processor(src, {'base':cumm}) if blend_processor else frame
  result = cumm + src * 0.0001
  result = result * 0.999
  return result

if __name__ == '__main__':
  # parse command line arguments
  parser = OptionParser()

  parser.add_option('-i', '--input', dest='input', default='saved-media/studio-test01-L-short.mp4')
  parser.add_option('--start-seconds', dest='start_seconds', default=None, type="float")
  parser.add_option('-c', '--config-file', dest='config_file', default='data/render_preprocess.json')
  parser.add_option('--config-file-blend', dest='config_file_blend', default='data/render_blend.json')
  parser.add_option('-v', '--verbose', dest="verbose", action='store_true', default=False)
  parser.add_option('--verbosity', dest="verbosity", action='store_true', default='info')

  opts, args = parser.parse_args()

  lvl = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}['debug' if opts.verbose else str(opts.verbosity).lower()]
  logging.basicConfig(level=lvl)
  #logger.setLevel({'debug': logging.DEBUG, 'info': logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}['debug' if opts.verbose else str(opts.verbosity).lower()])

  # print('Opening videocapture for: {}'.format(opts.input))
  iscam = opts.input.isdigit()
  capture = cv2.VideoCapture(int(opts.input) if iscam else opts.input)

  if type(opts.start_seconds != type(None)):
    fps = capture.get(cv2.CAP_PROP_FPS)
    startframe = int(opts.start_seconds * fps)
    # startframenum = start_framenum
    # if startframenum == None and start_time:
    #   # .get(cv2.CV_CAP_PROP_FRAME_COUNT)
    #   startframenum = int(float(start_time) * fps)

    if startframe:
      logger.info('Jumping to frame (start_seconds={} fps={}): {}'.format(opts.start_seconds, fps, startframe))
      capture.set(cv2.CAP_PROP_POS_FRAMES, startframe)

  # processor = create_processor_from_json_file(opts.config_file)
  pre_processor = create_controlled_processor_from_json_file(opts.config_file, winid='Pre-process-controls')
  blend_processor = create_controlled_processor_from_json_file(opts.config_file_blend, winid='blend-controls')
  image = np.zeros((720, 1280, 3), np.uint8)

  isDone = False
  try:
    while not isDone:
      # self.lastframe = s.cap.read()[1]
      ret,frame = capture.read()
      if ret:
        # self.lastframe = frame
        processed_frame = pre_processor(frame) if pre_processor else frame
        # cv2.imshow('pre-processed frame', processed_frame)

        image = render(image, processed_frame, blend_processor)
        cv2.imshow('render', image)

      elif not iscam:
        # loop
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

      # process user input
      key = cv2.waitKey(20) & 0xFF
      if key == 27 or key == ord('q'): # escape or Q
        isDone = True

      if key == ord('r'):
        # captureRef = True
        pre_processor = create_processor_from_json_file(opts.config_file)

      if isDone:
        break

  except KeyboardInterrupt:
    print('[process.py] KeyboardInterrupt, stopping.')
