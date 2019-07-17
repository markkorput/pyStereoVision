import os
# import numpy as np
import cv2
from optparse import OptionParser

DEFAULTS = {
  'video': None
}


def parse_args():
  parser = OptionParser()
  parser.add_option("-v", "--video", dest="video",
                    help="Video file to read from, (default: %default)",
                    default=DEFAULTS['video'])

  # parser.add_option("-o", "--out-file", dest="FILENAME",
  #                   help="Video file to write to", metavar="FILENAME",
  #                   default=DEFAULTS['filename'])

  # parser.add_option("-d", "--dimensions", dest="RES", type="choice",
  #                   choices=DEFAULTS['res_choices'],
  #                   default=DEFAULTS['res'],
  #                   help="Method to use. Valid choices are {}. Default: %default".format(DEFAULTS['res_choices'])) #, metavar="RES",)

  # parser.add_option("-f", "--fps", dest="FPS", type="float",
  #                   help="Frames Per Second", metavar="FPS",
  #                   default=DEFAULTS['fps'])

  # parser.add_option("-l", "--left", dest="CAM_L", type="int",
  #                   help="Camera ID for LEFT eye Camera", metavar="LEFT",
  #                   default=DEFAULTS['camL'])

  # parser.add_option("-r", "--right", dest="CAM_R", type="int",
  #                   help="Camera ID for RIGHT eye Camera", metavar="RIGHT",
  #                   default=DEFAULTS['camR'])

  # parser.add_option("-s", "--show",
  #                   action="store_false", dest="SHOW", default=DEFAULTS['show'],
  #                   help="Show recorded frames")

  (options, args) = parser.parse_args()
  return (options, args)

def get_stream(vid_path):
  cap = cv2.VideoCapture(vid_path)
  return {'cap': cap, 'ID': vid_path}

def get_LR_filenames(filename):
  filename, ext = os.path.splitext(filename)
  return (filename+'_L'+ext, filename+'_R'+ext)  

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
    while(True):
      update(streams)

      if cv2.waitKey(20) & 0xFF == ord('q'):
            break
  except KeyboardInterrupt:
    print("KeyboardInterrupt, stopping")
    
  for s in streams:
    s['cap'].release()

  cv2.destroyAllWindows()





