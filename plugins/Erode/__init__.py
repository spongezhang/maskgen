import cv2
from PIL import Image
import numpy

def transform(img,source,target,**kwargs):
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(numpy.asarray(img),kernel,iterations = 1)
    Image.fromarray(erosion).save(target)
    return None,None

# the actual link name to be used. 
# the category to be shown
def operation():
    return {'name':'Erode',
          'category':'Intensity',
          'description':'Erode one image with 5*5 kernel',
          'software':'OpenCV',
          'version':cv2.__version__,
          'arguments':None,
          'transitions': [
              'image.image'
          ]
          }

def suffix():
    return None
