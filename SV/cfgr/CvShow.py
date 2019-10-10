from cfgr.event import Event
import cv2

class CvShow:
  @staticmethod
  def cfgr(builder):
    ## outputs
    builder.addInput('id').string_to_method(lambda obj: obj.setId)
    builder.addInput('show').to_method(lambda obj: obj.showImage)
    builder.addInput('update').signal_to_method(lambda obj: obj.update)
    builder.addInput('destroyAllWindows').signal_to_method(lambda obj: cv2.destroyAllWindows)
    builder.addInput('waitKeyMs').int_to_method(lambda obj: obj.setWaitKeyMs)
    builder.addInput('verbose').bool_to_method(lambda obj: obj.setVerbose)

    builder.addOutput('key').from_event(lambda obj: obj.keyEvent)

  def __init__(self):
    self._id = ''
    self.waitKeyMs = 20
    self.keyEvent = Event()

  def setId(self, v): self._id = v
  def setWaitKeyMs(self, v): self.waitKeyMs = v

  def showImage(self, img):
    # height, width = img.shape[:2]
    # if height <= 0 or width <= 0:
    #   print('[CvShow] got zero-size frame to show')
    #   return
    cv2.imshow(self._id, img)

  def update(self):
    key = cv2.waitKey(self.waitKeyMs)
    if key != -1:
      self.verbose('[CvShow {}] key: {}'.format(self._id, key))
      self.keyEvent(key & 0xFF)

  def setVerbose(self, v): self.isVerbose = v
  def verbose(self, msg):
    if self.isVerbose:
      print(msg)