import imp
import os
import json
import subprocess

PluginFolder = os.path.join('.', "plugins")
MainModule = "__init__"

loaded = None

def getPlugins():
    plugins = {}
    possibleplugins = os.listdir(PluginFolder)
    customplugins = os.listdir(os.path.join(PluginFolder, 'Custom'))
    for i in possibleplugins:
        if i == 'Custom':
            continue
        location = os.path.join(PluginFolder, i)
        if not os.path.isdir(location) or not MainModule + ".py" in os.listdir(location):
            continue
        info = imp.find_module(MainModule, [location])
        plugins[i] = {"info": info}

    for j in customplugins:
        location = os.path.join(PluginFolder, 'Custom', j)
        plugins[os.path.splitext(j)[0]] = {"custom": location}

    return plugins

def loadPlugin(plugin):
    return imp.load_module(plugin['name'], *plugin["info"])

def loadCustom(plugin, path):
    """
    loads a custom plugin
    """
    global loaded
    print("Loading plugin " + plugin)
    with open(path) as jfile:
        data = json.load(jfile)
    loaded[plugin] = {}
    loaded[plugin]['function'] = 'custom'
    loaded[plugin]['operation'] = [data['operation']['name'],
                              data['operation']['category'],
                              data['operation']['description'],
                              data['operation']['softwarename'],
                              data['operation']['softwareversion']]
    loaded[plugin]['arguments'] = data['args'] if 'args' in data else None
    loaded[plugin]['command'] = data['command']
    loaded[plugin]['suffix'] = data['suffix'] if 'suffix' in data else None


def loadPlugins():
   global loaded
   if loaded is not None:
       return loaded

   loaded = {}
   ps = getPlugins() 
   for i in ps.keys():
      if 'custom' in ps[i]:
          path = ps[i]['custom']
          loadCustom(i, path)
      else:
          print("Loading plugin " + i)
          plugin = imp.load_module(MainModule, *ps[i]["info"])
          loaded[i] = {}
          loaded[i]['function'] = plugin.transform
          loaded[i]['operation'] = plugin.operation()
          loaded[i]['arguments'] = plugin.args()
          loaded[i]['suffix'] = plugin.suffix() if hasattr(plugin,'suffix') else None
   return loaded

def getOperations():
    global loaded
    ops = {}
    for l in loaded.keys():
        ops[l] = loaded[l]
    return ops

# return list of tuples, name and default value (which can be None)
def getArguments(name):
    global loaded
    return loaded[name]['arguments']

def getPreferredSuffix(name):
    global loaded
    return loaded[name]['suffix']

def getOperationNames(noArgs=False):
    global loaded
    if not noArgs:
      return loaded.keys()
    result = []
    for k,v in loaded.iteritems():
      if v['arguments'] is None or len(v['arguments'])==0:
        result.append(k)
    return result
    
def getOperation(name):
    global loaded
    return loaded[name]['operation']

def callPlugin(name,im,source,target,**kwargs):
    global loaded
    if loaded[name]['function'] == 'custom':
        return runCustomPlugin(name, im, source, target, **kwargs)
    else:
        return loaded[name]['function'](im,source,target,**kwargs)

def runCustomPlugin(name, im, source, target, **kwargs):
    global loaded
    executionCommand = loaded[name]['command'][:]
    for i in range(len(executionCommand)):
        if executionCommand[i] == '{inputimage}':
            executionCommand[i] = source
        elif executionCommand[i] == '{outputimage}':
            executionCommand[i] = target

        # Saved for once typed args are done...
        # else:
        #     executionCommand[i] = executionCommand[i].format(**kwargs)
    subprocess.call(executionCommand)
    return None, None
