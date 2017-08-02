from PIL import Image
import numpy as np
import cv2
import random
import os
from maskgen.image_wrap import ImageWrapper, openImageFile
from maskgen import tool_set,image_wrap
from skimage import  segmentation, color,measure,feature
from skimage.future import graph
import numpy as np
import math
from skimage.restoration import denoise_tv_bregman

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

from matplotlib import transforms
from matplotlib.patches import Ellipse

class TransformedEllipse(Ellipse):
    def __init__(self, xy, width, height, angle=0.0, fix_x=1.0, **kwargs):
        Ellipse.__init__(self, xy, width, height, angle, **kwargs)

        self.fix_x = fix_x

    def _recompute_transform(self):
        center = (self.convert_xunits(self.center[0]),
                  self.convert_yunits(self.center[1]))
        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)
        self._patch_transform = transforms.Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .rotate_deg(self.angle) \
            .scale(self.fix_x, 1) \
            .translate(*center)


def minimum_bounding_box(image):
    #(contours, _) = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    selected = []
    for cnt in contours:
        try:
            M = cv2.moments(cnt)
            x = int(M['m10'] / M['m00'])
            y = int(M['m01'] / M['m00'])
            x1, y1, w, h = cv2.boundingRect(cnt)
            #selected.append((w,h,w*h,x,y))
            selected.append((w,h,w*h,x,y))
        except:
            continue
        
    selected = sorted(selected, key=lambda cnt: cnt[2], reverse=True)
    
    if len(selected) == 0:
        print 'cannot determine contours'
        x, y, w, h = tool_set.widthandheight(image)
        selected.append((w,h,w*h,x,y))
    
    return selected[0]

def minimum_bounding_ellipse_of_points(points):
    M = cv2.moments(points)
    x = int(M['m10'] / M['m00'])
    y = int(M['m01'] / M['m00'])
    if x < 0 or y < 0 or len(points) < 4:
        return None
    if len(points) == 4:
        (x1, y1), (MA, ma), angle = cv2.minAreaRect(points)
    else:
        (x1, y1), (MA, ma), angle = cv2.fitEllipse(points)
    return (x, y, MA, ma, angle, cv2.contourArea(points), points)

def minimum_bounding_ellipse(image):
    """
    :param image:
    :return:  (x, y, MA, ma, angle, area, contour)
    @rtype : (int,int,int,int,float, float,np.array)
    """
    (contours, _) = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    selected = []
    for cnt in contours:
        try:
            bounded_info = minimum_bounding_ellipse_of_points(cnt)
            if bounded_info is not None:
               selected.append(bounded_info)
        except:
            continue
    if len(selected) == 0:
        return None
    selected = sorted(selected, key=lambda cnt: cnt[5], reverse=True)
    return selected[0] if len(selected) > 0 else None

def minimum_bounded_ellipse(image):
    image = feature.canny(image).astype(np.uint8)*255
    coords = np.column_stack(np.nonzero(image))
    # find the embedded circle using random sample consensus
    try:
       model, inliers = measure.ransac(coords, measure.EllipseModel,
                                    min_samples=3, residual_threshold=1,
                                    max_trials=500)
    except Exception as ex:
       print ex
       raise ex
    return model.params

def build_transform_matrix(placement_ellipse,mask_ellipse):
    width_diff = placement_ellipse[2]-mask_ellipse[2]
    height_diff = placement_ellipse[3] - mask_ellipse[3]
    scale_factor = 1.0 + (float(width_diff)/mask_ellipse[2])
    height_scale = 1.0 + (float(height_diff)/mask_ellipse[3])
    scale_factor = height_scale if height_scale < scale_factor else scale_factor
    if scale_factor < 0.2 or scale_factor > 1.5:
        return None
    rotation_angle = mask_ellipse[4]-placement_ellipse[4]
    return cv2.getRotationMatrix2D((mask_ellipse[0],mask_ellipse[1]),
                                   rotation_angle,scale_factor)

def transform_image(image,transform_matrix):
    """
    :param image:
    :param transform_matrix:
    :return:
    @rtype array
    """
    return cv2.warpAffine(image,transform_matrix,(image.shape[1],image.shape[0]),flags = cv2.WARP_INVERSE_MAP)

def build_random_transform(img_to_paste, mask_of_image_to_paste, image_center):
    scale = 0.5 + random.random()
    angle = 180.0*random.random() - 90.0
    return cv2.getRotationMatrix2D(image_center, angle, scale)

def pasteAnywhere(img, img_to_paste,mask_of_image_to_paste, simple):
    w, h, area, x, y = minimum_bounding_box(mask_of_image_to_paste)

    if not simple:
        rot_mat = build_random_transform(img_to_paste,mask_of_image_to_paste,(x,y))
        img_to_paste = cv2.warpAffine(img_to_paste, rot_mat, (img_to_paste.shape[1], img_to_paste.shape[0]))
        mask_of_image_to_paste= cv2.warpAffine(mask_of_image_to_paste, rot_mat, (img_to_paste.shape[1], img_to_paste.shape[0]))
        w, h, area, x, y = minimum_bounding_box(mask_of_image_to_paste)
        x, y, w1, h1 = tool_set.widthandheight(mask_of_image_to_paste)
        #x, y
    else:
        rot_mat = np.array([[1,0,0],[0,1,0]]).astype('float')
    
    if img.size[0] < w + 4:
        w = img.size[0] - 2
        xplacement = w/2 + 1
    else:
        xplacement = random.randint(w/2+1, img.size[0]-w/2-1)

    if img.size[1] < h + 4:
        h = img.size[1] - 2
        yplacement = h/2+1
    else:
        yplacement = random.randint(h/2+1, img.size[1]-h/2-1)

    #xplacement = random.randint(w/2+1, img.size[0]-w/2-1)
    #yplacement = random.randint(h/2+1,img.size[1]-h/2-1)
    output_matrix = np.eye(3, dtype=float)

    for i in range(2):
        for j in range(2):
            output_matrix[i,j] = rot_mat[i,j]
    output_matrix[0,2] = rot_mat[0,2] + xplacement - x - w1/2
    output_matrix[1,2] = rot_mat[1,2] + yplacement - y - h1/2

    return output_matrix, tool_set.place_in_image(
                          ImageWrapper(img_to_paste).to_mask().to_array(),
                          img_to_paste,
                          np.asarray(img),
                          (xplacement, yplacement),
                          rect = (x,y,w,h))

def transform(img,source,target,**kwargs):
    img_to_paste =openImageFile(kwargs['donor'])
    approach = kwargs['approach']  if 'approach' in kwargs else 'simple'
    segment_algorithm =  kwargs['segment'] if 'segment' in kwargs else 'felzenszwalb'
    mask_of_image_to_paste = img_to_paste.to_mask().to_array()
    out2 = None
    transform_matrix, out2 = pasteAnywhere(img, img_to_paste.to_array(), mask_of_image_to_paste, approach=='simple')
    ImageWrapper(out2).save(target)
    return {'transform matrix':tool_set.serializeMatrix(transform_matrix)} if transform_matrix is not None else None,None

# the actual link name to be used. 
# the category to be shown
def operation():
  return {'name':'AutoPasteSplice',
      'category':'Paste',
      'description':'Apply a mask to create an alpha channel',
      'software':'OpenCV',
      'version':'2.4.13',
      'arguments':{
          'donor':{
              'type':'donor',
              'defaultvalue':None,
              'description':'Mask to set alpha channel to 0'
          },
          'approach': {
              'type': 'list',
              'values': ['texture', 'simple', 'random'],
              'defaultvalue': 'random',
              'description': "The approach to find the placement. Option 'random' includes random selection scale and rotation"
          },
          'segment': {
              'type': 'list',
              'values' : ['felzenszwalb','slic'],
              'defaultvalue': 'felzenszwalb',
              'description': 'Segmentation algorithm for determiming paste region with simple set to no'
          }
      },
      'transitions': [
          'image.image'
      ]
  }

def suffix():
    return None
