from evento import Event


class Param:
  def __init__(self, name, default=None):
    self.name = name
    self.default = default
    self.value = default
    self.changeEvent = Event()

  def set(self, value):
    if self.value == value:
      return
    self.value = value
    self.changeEvent(self)

  def type(self):
    return type(self.value).__name__

class ParamsBuilder:
  def __init__(self, params):
    self.params = params

  def add(self, name, default=None):
    self.params.append(Param(name, default))

class Params(list):
  def __init__(self, name=''):
    self.name = name
    self.changeEvent = Event()

  def __enter__(self):
    self.builder = ParamsBuilder(self)
    return self.builder

  def __exit__(self, type, value, traceback):
    pass

  def get(self, name):
    return next(p for p in self if p.name == name)

  def append(self, param):
    list.append(self, param)
    def onParamChange(param):
      self.changeEvent(self)
    param.changeEvent += onParamChange

def toDict(params):
  d = {}
  for p in params:
    d[p.name] = p.value

  return d

def toSyncDict(params):
  d = toDict(params)

  def updatevalues(prms):
    newdict = toDict(prms)
    d.update(newdict)

  params.changeEvent += updatevalues
  return d

if __name__ == '__main__':
  params = Params('Test')
  with params as pars:
    pars.add('name', 'Bob')
    pars.add('delay', 0.05)
    pars.add('age', 8)

  print('Number of params: {}'.format(len(params)))
  print('Names: {}'.format(', '.join(map(lambda p: p.name, params))))
  print('Defaults: {}'.format(', '.join(map(lambda p: str(p.default), params))))

  values = toDict(params)
  print('Values: {}'.format(values))
  
  syncvals = toSyncDict(params)
  print('SyncValues: {}'.format(syncvals))
  
  params.get('name').set('John')
  print('SyncValues: {}'.format(syncvals))
