
from cfgr.event import Event
import cv2

class VideoCapture:
  STATE_UNSTARTED = 0
  STATE_STARTED = 1
  STATE_STOPPED = 2


  @staticmethod
  def cfgr(builder):
    ## outputs
    builder.addInput('id').int_to_method(lambda obj: obj.setCaptureId)
    builder.addInput('file').string_to_method(lambda obj: obj.setCaptureId)
    builder.addInput('resolution').list_to_method(lambda obj: obj.setResolutionFromList)
    builder.addInput('start').signal_to_method(lambda obj: obj.start)
    builder.addInput('stop').signal_to_method(lambda obj: obj.stop)
    builder.addInput('read').signal_to_method(lambda obj: obj.read)
    builder.addInput('verbose').bool_to_method(lambda obj: obj.setVerbose)

    builder.addOutput('frame').from_event(lambda obj: obj.frameEvent)

  def __init__(self):
    self.capId = 0
    self.resolution = None
    self.capture = None
    self.isVerbose = False
    self.state = VideoCapture.STATE_UNSTARTED
    self.frameEvent = Event()

  def read(self):
    if not self.capture:
      if self.state == VideoCapture.STATE_STOPPED:
        self.verbose('[VideoCapture {}] stopped, aborting read operation'.format(self.capId))
        return

      self.start()

    ret, frame = self.capture.read()

    if ret:
      # self.verbose('[VideoCapture id={}] emitting frame event with: {}'.format(self.capId, frame))
      self.frameEvent(frame)

  def start(self):
    self.verbose('[VideoCapture {}] starting'.format(self.capId))

    self.capture = cv2.VideoCapture(self.capId)

    if self.resolution:
      if type(self.resolution) == type([]) or type(self.resolution) == type(()):
        if len(self.resolution) == 2:
          self.verbose('[VideoCapture {}] setting resolution: {}x{}'.format(self.capId, self.resolution[0], self.resolution[1]))
          # Set resolution for the video capture
          # Function adapted from https://kirr.co/0l6qmh
          self.capture.set(3, self.resolution[0]) # width
          self.capture.set(4, self.resolution[1]) # height

    self.state = VideoCapture.STATE_STARTED

  def stop(self):
    if not self.capture:
      self.verbose('[VideoCapture {}] already stopped'.format(self.capId))
      return
    self.verbose('[VideoCapture {}] stopping'.format(self.capId))
    self.capture.release()
    self.capture = None
    self.state = VideoCapture.STATE_STOPPED

  def setResolutionFromList(self, v): self.resolution = v
  def setCaptureId(self, v): self.capId = v
  def setVerbose(self, v): self.isVerbose = v
  def verbose(self, msg):
    if self.isVerbose:
      print(msg)