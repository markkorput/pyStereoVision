from optparse import OptionParser
import cv2
from .utils.processor import create_controlled_processor_from_json_file, create_processor_from_json_file

if __name__ == '__main__':
  # parse command line arguments
  parser = OptionParser()

  parser.add_option('-i', '--input', dest='input', default='saved-media/studio-test01-L-short.mp4')
  parser.add_option('-c', '--config-file', dest='config_file', default='data/process.json')

  opts, args = parser.parse_args()

  # print('Opening videocapture for: {}'.format(opts.input))
  iscam = opts.input.isdigit()
  capture = cv2.VideoCapture(int(opts.input) if iscam else opts.input)

  # processor = create_processor_from_json_file(opts.config_file)
  processor = create_controlled_processor_from_json_file(opts.config_file, winid='Image Processors')

  isDone = False
  try:
    while not isDone:
      # self.lastframe = s.cap.read()[1]
      ret,frame = capture.read()
      if ret:
        # self.lastframe = frame
        processed_frame = processor(frame) if processor else frame
        cv2.imshow('processed frame', processed_frame)

      elif not iscam:
        # loop
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

      # process user input
      key = cv2.waitKey(20) & 0xFF
      if key == 27 or key == ord('q'): # escape or Q
        isDone = True

      if key == ord('r'):
        # captureRef = True
        processor = create_processor_from_json_file(opts.config_file)

      if isDone:
        break

  except KeyboardInterrupt:
    print('[process.py] KeyboardInterrupt, stopping.')
