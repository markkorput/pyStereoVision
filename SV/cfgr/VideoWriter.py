from cfgr.event import Event
import cv2, os.path

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPES = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
  filename, ext = os.path.splitext(filename)
  if ext in VIDEO_TYPES:
    return VIDEO_TYPES[ext]
  return VIDEO_TYPES['avi']

class VideoWriter:
  STATE_UNSTARTED = 0
  STATE_STARTED = 1
  STATE_STOPPED = 2

  @staticmethod
  def cfgr(builder):
    ## outputs
    builder.addInput('to').string_to_method(lambda obj: obj.setTarget)
    builder.addInput('fps').float_to_method(lambda obj: obj.setFps)
    builder.addInput('resolution').list_to_method(lambda obj: obj.setResolution)
    builder.addInput('codec').string_to_method(lambda obj: obj.setCodec)
    builder.addInput('frame').to_method(lambda obj: obj.writeFrame)
    builder.addInput('stop').signal_to_method(lambda obj: obj.stop)
    builder.addInput('verbose').bool_to_method(lambda obj: obj.setVerbose)

    builder.addOutput('frameWritten').from_event(lambda obj: obj.frameWrittenEvent)

  def __init__(self):
    self.target = "recording.avi"
    self.fps = 24.0
    self.resolution = [640,480]
    self.codec = None
    self.writer = None
    self.isVerbose = False
    self.state = VideoWriter.STATE_UNSTARTED
    self.frameWrittenEvent = Event()

  # def fire(self, *args, **kwargs):
  #   self.doEvent(*args, **kwargs)

  def writeFrame(self, frame):
    if not self.writer:
      if self.state == VideoWriter.STATE_STOPPED:
        self.verbose('[VideoWriter {}] frame rejected because stopped'.format(self.target))
        return
      self.start()

    frame_res = frame.shape[:2]
    if frame_res[0] != self.resolution[1] or frame_res[1] != self.resolution[0]:
      self.verbose("[VideoWriter {}] got frame size ({}) that differs from write resolution ({})".format(self.target, frame_res, self.resolution))
    self.writer.write(frame)
    # self.verbose("[VideoWriter {}] frame written: {} lines".format(self.target, len(frame)))
    self.frameWrittenEvent(frame)

  def start(self):
    res = self.resolution if self.resolution and len(self.resolution) == 2 else [640,480]
    self.verbose("[VideoWriter {}] starting (resolution: {}x{})".format(self.target, res[0], res[1]))
    self.writer = cv2.VideoWriter(self.target, get_video_type(self.target), self.fps, res)
    self.state = VideoWriter.STATE_STARTED

  def stop(self):
    if not self.writer:
      self.verbose("[VideoWriter {}] already stopped".format(self.target))
      return
    self.verbose("[VideoWriter {}] stopping".format(self.target))
    self.writer.release()
    self.stoppedWriter = self.writer
    self.writer = None
    self.state = VideoWriter.STATE_STOPPED
  

  def setResolution(self, v): self.resolution = v if type(v) == type(()) else tuple(v)
  def setCodec(self, v): self.codec = v
  def setFps(self, v): self.fps = v
  def setTarget(self, v): self.target = v
  def setVerbose(self, v): self.isVerbose = v
  def verbose(self, msg):
    if self.isVerbose:
      print(msg)