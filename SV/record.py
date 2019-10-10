# original logic taken from:
# https://github.com/codingforentrepreneurs/OpenCV-Python-Series/blob/master/src/lessons/record-video.py

import os
# import numpy as np
import cv2
from optparse import OptionParser

DEFAULTS = {
  'filename': 'saved-media/video.avi', # .avi .mp4
  'fps': 24.0,
  'res': '720p', # 480p | 720p | 1080p | 4k
  'res_choices': ('480p', '720p', '1080p', '4k'),
  'camL': 0,
  'camR': 1,
  'show': True
}

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height



# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
  filename, ext = os.path.splitext(filename)
  if ext in VIDEO_TYPE:
    return VIDEO_TYPE[ext]
  return VIDEO_TYPE['avi']



def get_LR_filenames(filename):
  filename, ext = os.path.splitext(filename)
  return (filename+'_L'+ext, filename+'_R'+ext)  

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option("-o", "--out-file", dest="FILENAME",
                    help="Video file to write to", metavar="FILENAME",
                    default=DEFAULTS['filename'])

  parser.add_option("-d", "--dimensions", dest="RES", type="choice",
                    choices=DEFAULTS['res_choices'],
                    default=DEFAULTS['res'],
                    help="Method to use. Valid choices are {}. Default: %default".format(DEFAULTS['res_choices'])) #, metavar="RES",)

  parser.add_option("-f", "--fps", dest="FPS", type="float",
                    help="Frames Per Second", metavar="FPS",
                    default=DEFAULTS['fps'])

  parser.add_option("-l", "--left", dest="CAM_L", type="int",
                    help="Camera ID for LEFT eye Camera", metavar="LEFT",
                    default=DEFAULTS['camL'])

  parser.add_option("-r", "--right", dest="CAM_R", type="int",
                    help="Camera ID for RIGHT eye Camera", metavar="RIGHT",
                    default=DEFAULTS['camR'])

  parser.add_option("-s", "--show",
                    action="store_false", dest="SHOW", default=DEFAULTS['show'],
                    help="Show recorded frames")

  (options, args) = parser.parse_args()

  save_path_L, save_path_R = get_LR_filenames(options.FILENAME)

  print("Config:")  
  print("Cameras (left/right): {}/{}".format(options.CAM_L, options.CAM_R))
  print("Out files (left/right): {}, {}".format(save_path_L, save_path_R))
  print("Resolution: {}".format(options.RES))
  print("FPS: {}".format(options.FPS))

  def get_stream(camId, res, fps, filepath):
    cap = cv2.VideoCapture(camId)
    dims = get_dims(cap, res=res)
    video_type_cv2 = get_video_type(filepath)
    writer = cv2.VideoWriter(filepath, video_type_cv2, fps, dims)
    return {'cap': cap, 'dims': dims, 'vidtype': video_type_cv2, 'writer': writer, 'ID': filepath}

  streams = []
  if not options.CAM_L == -1:
    streams.append(
      get_stream(options.CAM_L,
        options.RES,
        options.FPS,
        save_path_L))

  if not options.CAM_R == -1:
    streams.append(
      get_stream(options.CAM_R,
        options.RES,
        options.FPS,
        save_path_R))

  counter = 0

  print("Starting recording, press 'Q' or CTRL+C to stop...")
  try:
    while(True):
        # Capture frame-by-frame
        # ret, frame = capL.read()
        for s in streams:
          s['lastframe'] = s['cap'].read()[1]

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # outL.write(frame)
        for s in streams:
          s['writer'].write(s['lastframe'])
          print("Wrote {} lines".format(len(s['lastframe'])))

        counter += 1
        print ("Recorded frame {}".format(counter))

        # Display the resulting frame
        if options.SHOW:
          for s in streams:
            cv2.imshow(s['ID'], s['lastframe'])

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

  except KeyboardInterrupt:
    print("KeyboardInterrupt, stopping")


  # When everything done, release the capture
  # cap.release()
  # out.release()
  for s in streams:
    s['cap'].release()
    s['writer'].release()

  cv2.destroyAllWindows()