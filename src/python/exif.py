from subprocess import call,Popen, PIPE
import numpy as np
from PIL import Image
import os

def getOrientationFromExif(source):
    orientations= [None, None,'Mirror horizontal','Rotate 180','Mirror vertical','Mirror horizontal and Rotate 270 CW','Rotate 90 CW','Mirror horizontal and rotate 90 CW','Rotate 270 CW']

    exifcommand = os.getenv('MASKGEN_EXIFTOOL','exiftool')
    rotateStr = Popen([exifcommand, '-n', '-Orientation', source],
                        stdout=PIPE).communicate()[0]

    rotation = rotateStr.split(':')[1].strip() if rotateStr.rfind(':') > 0 else '-'

    if rotation == '-':
        return None
    try:
      rotation_index = int(rotation)
      return orientations[rotation_index]
    except:
      return None

def rotateAccordingToExif(im, orientation):
    rotation = orientation

    if rotation is None:
        return im
    arr = np.array(im)
    if rotation == 'Mirror horizontal':
        rotatedArr = np.fliplr(arr)
    elif rotation == 'Rotate 180':
        rotatedArr = np.rot90(arr,2)
    elif rotation == 'Mirror vertical':
        rotatedArr = np.flipud(arr)
    elif rotation == 'Mirror horizontal and rotate 270 CW':
        rotatedArr = np.fliplr(arr)
        rotatedArr = np.rot90(rotatedArr,3)
    elif rotation == 'Rotate 90 CW':
        rotatedArr = np.rot90(arr)
    elif rotation == 'Mirror horizontal and rotate 90 CW':
        rotatedArr = np.fliplr(arr)
        rotatedArr = np.rot90(rotatedArr)
    elif rotation == 'Rotate 270 CW':
        rotatedArr = np.rot90(arr, 3)
    else:
        rotatedArr = arr

    rotatedIm = Image.fromarray(rotatedArr)
    return rotatedIm


def copyexif(source,target):
  exifcommand = os.getenv('MASKGEN_EXIFTOOL','exiftool')
  try:
     call([exifcommand, '-all=', target])
     call([exifcommand, '-P', '-TagsFromFile',  source, '-all:all', '-unsafe', target])
     call([exifcommand, '-XMPToolkit=', target])
     call([exifcommand, '-Warning=', target])
     return None
  except OSError:
     return 'exiftool not installed'

def getexif(source):
  exifcommand = os.getenv('MASKGEN_EXIFTOOL','exiftool')
  meta = {}
  try:
    p = Popen([exifcommand,source],stdout=PIPE,stderr=PIPE)
    try:
      while True:
        line = p.stdout.readline()
        try:
           line = unicode(line,'utf-8')
        except:
           try:
             line = unicode(line,'latin').encode('ascii',errors='xmlcharrefreplace')
           except:
             continue
        if line is None or len(line) == 0:
           break
        pos = line.find(': ')
        if pos > 0:
           meta[line[0:pos].strip()] = line[pos+2:].strip()
    finally:
      p.stdout.close()
      p.stderr.close()
  except OSError:
    print "Exiftool not installed"
  return meta

def _decodeStr(name,sv):
  return sv
       
def compareexif(source,target):
  metasource = getexif(source)
  metatarget = getexif(target)
  diff = {}
  for k,sv in metasource.iteritems():
     if k in metatarget:
       tv = metatarget[k]
       if tv != sv:
         diff[k] = ('change',sv,tv)
     else:
         diff[k] = ('delete',sv)
  for k,tv in metatarget.iteritems():
     if k not in metasource:
         diff[k] = ('add',tv)
  return diff

