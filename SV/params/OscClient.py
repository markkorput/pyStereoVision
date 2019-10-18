from evento import Event
import socket, logging

DEPS = None # {'osc_message_builder': None, 'dispatcher': None}

def loadDeps():
  result = {}

  def loadEmbeddeDeps(result):
    try:
      from cfgr.embeds.oscpy.client import OSCClient
      result['OSCClient'] = OSCClient
    except ImportError:
      # logging.getLogger(__name__).warning("failed to load pythonosc dependency; OscOout component will not work")
      pass

  try:
    from pythonosc import udp_client
    result['udp_client'] = udp_client
  except ImportError:
    loadEmbeddeDeps(result)
  except Exception:
    loadEmbeddeDeps(result)

  return result

class OscClient:
  """
  Osc sender client
  """
  def __init__(self, host, port):
    self.host = None
    self.port = 0
    self.client = None

    self.connectEvent = Event()
    self.disconnectEvent = Event()
    # self.messageEvent = Event()

  def __del__(self):
    self.disconnect()

  def isConnected(self): return self.client != None

  def connect(self):
    global DEPS
    host = self.host
    port = self.port

    if not host:
      print("[OscClient] no host, can't connect")
      return False
    
    if DEPS == None:
      DEPS = loadDeps()

      if not 'udp_client' in DEPS and not 'OSCClient' in DEPS:
        print("[OscClient] missing OSC dependency")

    if not 'udp_client' in DEPS and not 'OSCClient' in DEPS:
      return False

    # try:
    #     # self.client = OSC.OSCClient()
    #     # if target.endswith('.255'):
    #     #     self.logger.info('broadcast target detected')
    #     #     self.client.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # except OSC.OSCClientError as err:
    #     self.logger.error("OSC connection failure: {0}".format(err))
    #     return False

    if 'udp_client' in DEPS:
      self.client = DEPS['udp_client'].SimpleUDPClient(host, port)
      self.connected = True
      self.connectEvent(self)
      self.verbose('[OscClient {}:{}] connected'.format(self.host, self.port))

    if 'OSCClient' in DEPS:
      self.client = DEPS['OSCClient'](host, port)
      self.connected = True
      self.connectEvent(self)
      self.verbose('[OscClient {}:{}] connected using OSCClient'.format(self.host, self.port))

    return True

  def disconnect(self):
    if not self.isConnected():
      return

    # self.client.close()
    self.client = None
    self.disconnectEvent(self)
    self.verbose('[OscClient {}:{}] disconnected'.format(self.host, self.port))

  def send(self, addr, args):
    if not self.isConnected():
      if not self.connect():
        print('[OscClient {}:{}] failed to connect; could not send message {} [{}]'.format(self.host, self.port, addr, ", ".join(map(lambda x: str(x), args))))
        return
    
    if 'OSCClient' in DEPS:
      try:
        self.client.send_message(addr.encode('ascii'), args)
        # self.messageEvent((addr,args))
        logging.debug('[OscClient {}:{}] sent {} using OSCClient [{}]'.format(self.host, self.port, addr, ", ".join(map(lambda x: str(x), args))))
      except AttributeError as err:
        print('[OscClient {}:{}] send error: {}'.format(self.host, self.port, str(err)))
      except socket.gaierror as err:
        print('[OscClient {}:{}] failed send message {} [{}]: {}'.format(self.host, self.port, addr, ", ".join(map(lambda x: str(x), args)), str(err)))
      return

    try:
      self.client.send_message(addr, args)
      # self.messageEvent((addr,args))
      logging.debug('[OscClient {}:{}] sent {} [{}]'.format(self.host, self.port, addr, ", ".join(map(lambda x: str(x), args))))
    #     # self.client.send(msg)s
    # except OSC.OSCClientError as err:
    #     pass
    except AttributeError as err:
      print('[OscClient {}:{}] send error: {}'.format(self.host, self.port, str(err)))
    except socket.gaierror as err:
      print('[OscClient {}:{}] failed send message {} [{}]: {}'.format(self.host, self.port, addr, ", ".join(map(lambda x: str(x), args)), str(err)))