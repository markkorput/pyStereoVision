import os, time
# import numpy as np
import cv2
from optparse import OptionParser

DEFAULTS = {
  'video': None,
  'delay': None
}

def parse_args():
  parser = OptionParser()
  parser.add_option("-v", "--video", dest="video",
                    help="Video file to read from, (default: %default)",
                    default=DEFAULTS['video'])

  parser.add_option("-d", "--delay", dest="delay", type="float",
                    help="Delay between each frame",
                    default=DEFAULTS['delay'])

  (options, args) = parser.parse_args()
  return (options, args)

def get_LR_filenames(filename):
  filename, ext = os.path.splitext(filename)
  return (filename+'_L'+ext, filename+'_R'+ext)  

def get_stream(vid_path):
  cap = cv2.VideoCapture(vid_path)
  return {'cap': cap, 'ID': vid_path}

def update(streams):
  for s in streams:
    (retval, frame) = s['cap'].read()
    if retval:
      cv2.imshow(s['ID'], frame)

if __name__ == '__main__':
  (options, args) = parse_args()
  print(options)

  streams = []

  if options.video:
    save_path_L, save_path_R = get_LR_filenames(options.video)
    print('Using videos: {}, {}'.format(save_path_L,save_path_R))
    streams.append(get_stream(save_path_L))
    streams.append(get_stream(save_path_R))

  print("Starting calibration, press 'Q' or CTRL+C to stop...")
  try:
    nextFrameTime = time.time()
    while(True):
      if not options.delay or time.time() > nextFrameTime:
        update(streams)
        nextFrameTime = time.time() + options.delay

      if cv2.waitKey(20) & 0xFF == ord('q'):
            break
  except KeyboardInterrupt:
    print("KeyboardInterrupt, stopping")
    
  for s in streams:
    s['cap'].release()

  cv2.destroyAllWindows()





