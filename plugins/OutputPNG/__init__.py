from PIL import Image
from maskgen import exif
import numpy as np

"""
Save te image as PNG. If the image has a orientation and 'Image Rotated', rotate the image according to the EXIF.
"""
def transform(img,source,target, **kwargs):
    im = Image.open(source)
    im = np.array(im)
    if len(im.shape)==2:
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, :] = im[:, :, np.newaxis]
        im = ret
    im = Image.fromarray(im)

    if 'Image Rotated' in kwargs and kwargs['Image Rotated'] == 'yes':
        orientation = exif.getOrientationFromExif(source)
        if orientation is not None:
            im = Image.fromarray(exif.rotateAccordingToExif(np.asarray(im),orientation, counter=True))
    im.save(target,format='PNG')
    
    return None,None
    
def operation():
    return {'name':'OutputPng',
            'category':'Output',
            'description':'Save an image as .PNG',
            'software':'PIL',
            'version':'1.1.7',
            'arguments':{
                'Image Rotated':{
                    'type':'yesno',
                    'defaultvalue':'no',
                    'description':'Rotate image according to EXIF'
                }
            },
            'transitions': [
                'image.image'
            ]
        }

def suffix():
    return '.png'
