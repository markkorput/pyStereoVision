# original logic taken from:
# https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/lessons/record-video.py

import os, logging, cv2, time, math
from optparse import OptionParser
from .utils import isNone

DEFAULTS = {
  'input': "0,1",
  'output': 'saved-media/recording-L.avi,saved-media/recording-R.avi',

  # 'filename': 'saved-media/video.avi', # .avi .mp4
  'fps': 24.0,
  'res': '720p', # 480p | 720p | 1080p | 4k
  'res_choices': ('480p', '720p', '1080p', '4k'),
  # 'camL': 0,
  # 'camR': 1,
  'show': True
}

# A stream is an input and and output
class Stream:
  def __init__(self, input, output, res, fps):
    self.input = input
    self.output = output
    self.res = res
    self.fps = fps

    self.cap = cv2.VideoCapture(int(self.input) if self.input.isdigit() else self.input)

    self.writer = None
    self.lastframe = None
    self.firstWriteTime = None
    self.writeFrameCount = 0

  def __del__(self):
    self.stopRec()

    if self.cap:
      logging.debug("Capture for {}".format(self.input))
      self.cap.release()
      self.cap = None

  def startRec(self):
    STD_DIMENSIONS =  {
      "480p": (640, 480),
      "720p": (1280, 720),
      "1080p": (1920, 1080),
      "4k": (3840, 2160)
    }

    self.dimms = STD_DIMENSIONS['480p']
    if self.res in STD_DIMENSIONS:
        self.dimms = STD_DIMENSIONS[self.res]

    self.cap.set(3, self.dimms[0])
    self.cap.set(4, self.dimms[1])

    # Video Encoding, might require additional installs
    # Types of Codes: http://www.fourcc.org/codecs.php
    VIDEO_TYPE = {
        'avi': cv2.VideoWriter_fourcc(*'XVID'),
        #'mp4': cv2.VideoWriter_fourcc(*'H264'),
        'mp4': cv2.VideoWriter_fourcc(*'XVID'),
    }

    filename, ext = os.path.splitext(self.output)
    typ = VIDEO_TYPE[ext] if ext in VIDEO_TYPE else VIDEO_TYPE['avi']
    self.writer = cv2.VideoWriter(self.output, typ, self.fps, self.dimms)

  def stopRec(self):
    if self.writer:
      logging.debug("Releasing writer for {}".format(self.output))
      self.writer.release()
      self.writer = None

  def recIsStarted(self):
    return not self.writer == None

  def read(self):
    # self.lastframe = s.cap.read()[1]
    ret,frame = self.cap.read()
    if ret:
      self.lastframe = frame

  def write(self, grayscale=False, sync=True):
    if not self.writer:
      return

    frame = self.lastframe
    # Convert to grayscale?
    if grayscale:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # write frame
    self.writer.write(frame)
    self.writeFrameCount += 1

    if sync:
      if not self.firstWriteTime:
        self.firstWriteTime = time.time()
        return

      rectime = time.time() - self.firstWriteTime
      syncFrameCount = rectime * self.fps

      while self.writeFrameCount < syncFrameCount:
        logging.debug('Writing padding frame for syncing')
        self.writer.write(frame)
        self.writeFrameCount += 1

def main(input="0,1", output="saved-media/record-L.avi,saved-media/record-R.avi", res='720p', fps=25, grayscale=False, show=True, verbose=False, sync=False, startRecording=True):
  logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format='%(asctime)s %(message)s')

  ##### PRE-CHECK #####

  inputs = input.split(',')
  outputs = output.split(',')

  if len(inputs) != len(outputs):
    logging.warning('Number of inputs ({}: {}) doesn\'t match with number of outputs ({}: {})'.format(len(inputs), inputs, len(outputs), outputs))
    return

  ##### INIT #####

  streams = []

  for idx, input in enumerate(inputs):
    stream = Stream(input, outputs[idx], res, fps)
    streams.append(stream)
    if startRecording:
      stream.startRec()

  ##### RECORD #####

  counter = 0

  logging.info("Starting recording, press <ESC>, 'Q' or CTRL+C to stop...")
  try:
    while(True):
        ## READ ##
        for s in streams:
          s.read()

        ## WRITE ##
        for s in streams:
          s.write(grayscale=grayscale, sync=sync)

        counter += 1
        print("Number of frames recorded: {}\r".format(counter), end='')

        # Display the resulting frame
        if show:
          for s in streams:
            if not isNone(s.lastframe):
              cv2.imshow(s.input, s.lastframe)

        key = cv2.waitKey(3) & 0xFF

        if key != -1 and key != 255:
          if key == ord('q') or key == 27:
            break
          if key == ord('r'):
            for s in streams:
              if s.recIsStarted():
                s.stopRec()
              else:
                s.startRec()

  except KeyboardInterrupt:
    logging.debug("KeyboardInterrupt, stopping")

  logging.info('Total number of frames recorded: {}'.format(counter))

  ##### CLEANUP #####

  cv2.destroyAllWindows()

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option("-o", "--output", dest="output",
                    help="Video file(s) to write to", metavar="FILENAMES",
                    default=DEFAULTS['output'])

  parser.add_option("-i", "--input", dest="input",
                    help="Video stream(s) to read from to", metavar="FILENAMES",
                    default=DEFAULTS['input'])

  # parser.add_option("-l", "--left", dest="CAM_L", type="int",
  #                   help="Camera ID for LEFT eye Camera", metavar="LEFT",
  #                   default=DEFAULTS['camL'])

  # parser.add_option("-r", "--right", dest="CAM_R", type="int",
  #                   help="Camera ID for RIGHT eye Camera", metavar="RIGHT",
  #                   default=DEFAULTS['camR'])

  parser.add_option("-d", "--dimensions", dest="RES", type="choice",
                    choices=DEFAULTS['res_choices'],
                    default=DEFAULTS['res'],
                    help="Method to use. Valid choices are {}. Default: %default".format(DEFAULTS['res_choices'])) #, metavar="RES",)

  parser.add_option("-f", "--fps", dest="FPS", type="float",
                    help="Frames Per Second", metavar="FPS",
                    default=DEFAULTS['fps'])


  parser.add_option("-s", "--show",
                    action="store_false", dest="SHOW", default=DEFAULTS['show'],
                    help="Show recorded frames")

  parser.add_option("-g", "--grayscale",
                    action="store_true", dest="grayscale", default=False,
                    help="Record grayscale")

  parser.add_option("-t", "--time-sync",
                    action="store_true", dest="sync", default=False,
                    help="Record additional frames if time syncing requires it")

  parser.add_option("-R", "--no-rec",
                    action="store_true", dest="norec", default=False,
                    help="Don't auto-start recording")

  parser.add_option("-v", "--verbose",
                    action="store_true", dest="verbose", default=False,
                    help="Verbose logging")

  (opts, args) = parser.parse_args()

  main(input=opts.input, output=opts.output, res=opts.RES, fps=opts.FPS, show=opts.SHOW, grayscale=opts.grayscale, verbose=opts.verbose, sync=opts.sync, startRecording=not opts.norec)
