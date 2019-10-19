from .Params import Params, toDict
import threading, logging, re, json
from evento import Event
from .OscClient import OscClient

DEPS = None # {'osc_server': None, 'dispatcher': None}

def loadDeps():
  result = {}
  try:
    from pythonosc import dispatcher
    from pythonosc import osc_server
    result['osc_server'] = osc_server
    result['dispatcher'] = dispatcher
  except ImportError:
    from cfgr.embeds.python2osc import dispatcher
    from cfgr.embeds.python2osc import osc_server
    result['osc_server'] = osc_server
    result['dispatcher'] = dispatcher
  except Exception:
    from cfgr.embeds.python2osc import dispatcher
    from cfgr.embeds.python2osc import osc_server
    result['osc_server'] = osc_server
    result['dispatcher'] = dispatcher

    # logging.getLogger(__name__).warning("failed to load pythonosc dependency; OscServer component will not work")
    pass

  return result

class OscServer:
  def __init__(self, port=8080, host=''):
    self.host = host
    self.port = port
    self.server = None
    self.thread = None
    self.isConnected = False

    self.connectedEvent = Event()
    self.disconnectedEvent = Event()
    self.messageEvent = Event()

  def __del__(self):
    self.stop()

  def start(self):
    global DEPS

    if self.isConnected:
      return False

    if DEPS == None:
      DEPS = loadDeps()

      if not 'osc_server' in DEPS or not 'dispatcher' in DEPS:
        print('[OscServer] could not load dependencies')

    if not 'osc_server' in DEPS or not 'dispatcher' in DEPS:
      return False

    disp = DEPS['dispatcher'].Dispatcher()
    disp.map("*", self._onOscMsg)
    # disp.map("/logvolume", print_compute_handler, "Log volume", math.log)

    result = False
    try:
      self.server = DEPS['osc_server'].ThreadingOSCUDPServer((self.host, self.port), disp)
      self.server.daemon_threads = True
      self.server.timeout = 1.0
      # self.server = DEPS['osc_server'].BlockingOSCUDPServer((self.host, self.port), disp)
      def threadFunc():
          try:
              self.server.serve_forever()
          except KeyboardInterrupt:
              pass
          self.server.server_close()

      self.thread = threading.Thread(target = threadFunc);
      self.thread.start()
      # set internal connected flag
      result = True
    except OSError as err:
      print("[OscServer] Could not start OSC server: "+str(err))

    if result:
      # notify
      logging.debug("[OscServer {0}:{1}] server started".format(self.host, str(self.port)))
      self.connectedEvent(self)

    self.isConnected = result
    return result

  def stop(self):
    if self.isConnected:
      if self.server:
        self.server._BaseServer__shutdown_request = True
        self.server.shutdown()
        self.server = None
      self.isConnected = False
      self.disconnectedEvent(self)
      logging.debug('[OscServer {0}:{1}] server stopped'.format(self.host, str(self.port)))

  def _onOscMsg(self, addr, *args):
    logging.debug('[OscServer {0}:{1}] received {2} [{3}]'.format(self.host, self.port, addr, ", ".join(map(lambda x: str(x), args))))
    self.messageEvent(addr, args)

class AddrParser:
  def __init__(self, addr, scope=''):
    self.addr = addr
    self.scope = scope
  
  def unscoped(self):
    return re.compile('^{}'.format(self.scope)).sub('', self.addr)

  def action(self):
    action = self.unscoped().split('/')[1]
    # logging.info('Action: {}'.format(action))
    return action

  def path(self):
    path = '/'+'/'.join(self.unscoped().split('/')[2:])
    # logging.info('Path: {}'.format(path))
    return path

class OscListener:
  def __init__(self, paramsList, port=8081, scope="/PyParams", start=True):
    self.paramsList = [paramsList] if type(paramsList) == type(Params()) else paramsList
    self.scope = scope
    self.server = OscServer(port)
    self.server.messageEvent += self.onMessage

    if start:
      self.server.start()

  def __del__(self):
    self.server.messageEvent -= self.onMessage
    self.server.stop()

  def onMessage(self, addr, args):
    parser = AddrParser(addr, scope=self.scope)
    logging.debug('got {} message with: {}'.format(parser.action(), args))

    # set param value message?
    # "/scope/set/Test/name", ('John')
    if parser.action() == 'set':
      path = parser.path()
      p = self.getParamForPath(path)
      if not p:
        logging.info('Got invalid set path: {}'.format(addr))
        return
      
      if len(args) < 1:
        logging.info('Did not get value argument with OSC message {}'.format(addr))
        return

      value = args[0]
      logging.debug('Setting {} to {} (osc: {} {})'.format(p.name, value, addr, args))
      p.set(value)
      return

    # "/scope/info, ('127.0.0.1', 8085)
    if parser.action() == 'info' or parser.action() == 'signup':
      if len(args) < 3:
        logging.warning('Received `info` OSC message with less than 3 arguments, ignoring')
        return

      host = args[0]
      port = args[1]
      addr = args[2]
      json = self.getInfoJson()

      # response client
      client = OscClient(host, port)
      client.send(addr, json)

  def getParamForPath(self, path):
    for params in self.paramsList:
      if path.startswith('/{}/'.format(params.name)):
        parname = re.compile('^/{}/'.format(params.name)).sub('', path)
        return params.get(parname)

    return None

  def parse(self, addr):
    parse = {}
    parse['unscoped'] = re.compile('^{}'.format(self.scope)).sub('', addr)
    parse['action'] = parse['unscoped'].split('/')[0]
    parse['path'] = '/'.join(parse['unscoped'].split('/')[1:])

  def getInfoJson(self):
    items = []
    for params in self.paramsList:
      for p in params:
        items.append({
          'name': p.name,
          'type': p.type(),
          'default': p.default,

          'setAddr': '{}/set/{}/{}'.format(self.scope, params.name, p.name)
        })
    
    return json.dumps({
      'params': items
    })


if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

  params = Params('Test')
  with params as pars:
    pars.add('name', 'Bob')
    pars.add('age', 21)
    pars.add('score', 64.05)


  oscServer = OscListener(params, 8010, scope='/FooBar', start=False)
  print('Initial: {}'.format(toDict(params)))
  oscServer.onMessage('/FooBar/set/Test/name', ['John'])
  print('After name change: {}'.format(toDict(params)))
  oscServer.onMessage('/FooBar/set/Test/age', [33])
  print('After age change: {}'.format(toDict(params)))

  print('metadata.json: {}'.format(oscServer.getInfoJson()))

  listener = OscListener(params)

  while True:
    pass