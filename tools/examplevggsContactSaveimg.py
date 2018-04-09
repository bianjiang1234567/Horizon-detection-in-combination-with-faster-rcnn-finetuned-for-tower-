from __future__ import division
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, '..', 'caffe', 'python')
add_path(caffe_path)


import caffe
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def center_crop(im):
  
  sz = im.shape[0:-1]
  side_length = np.min(sz)
  if sz[0] > sz[1]:
    ul_x = 0  
    ul_y = np.floor((sz[0]/2) - (side_length/2))
    x_inds = [ul_x, sz[1]-1]
    y_inds = [ul_y, ul_y+side_length-1]
  else:
    ul_x = np.floor((sz[1]/2) - (side_length/2))
    ul_y = 0
    x_inds = [ul_x, ul_x+side_length-1]
    y_inds = [ul_y, sz[0]-1]

  c_im = im[int(y_inds[0]):int(y_inds[1]+1), int(x_inds[0]):int(x_inds[1]+1), :]

  return c_im, [c_im.shape, x_inds, y_inds]

def preprocess(im, caffe_sz):
  
  # scale to [0, 255] 
  im = (255*im).astype(np.float32, copy=False)

  # channel swap (RGB -> BGR)
  im = im[:, :, [2,1,0]]
  
  # make caffe size
  im = caffe.io.resize_image(im, caffe_sz)

  # subtract mean 
  #im = im - np.asarray([104,117,123]).reshape((1,1,3))
  #im = im - np.asarray([116.377,115.814,117.027]).reshape((1,1,3))
  im = im - np.asarray([115.894,117.165,118.294]).reshape((1,1,3))
  
  # make channels x height x width
  im = im.swapaxes(0,2).swapaxes(1,2)
 
  caffe_input = im.reshape((1,)+im.shape)
 
  return caffe_input

def bin2val(bin_id, bin_edges):
  assert 0 <= bin_id < bin_edges.size-1, 'impossible bin_id'

  # handle infinite bins, choose left/right edge as appropriate
  if bin_id == 0 and bin_edges[0] == -np.inf:
    val = bin_edges[1]
  elif bin_id == bin_edges.size-2 and bin_edges[-1] == np.inf:
    val = bin_edges[-2]
  else:
    val = (bin_edges[bin_id] + bin_edges[bin_id+1]) / 2

  return val

def extrap_horizon(left, right, width):
  
  hl_homo = np.cross(np.append(left, 1), np.append(right, 1))
  hl_left_homo = np.cross(hl_homo, [-1, 0, -width/2]);
  #hl_left_homo = np.cross(hl_homo, [-1, 0, -5000]);
  hl_left = hl_left_homo[0:2]/hl_left_homo[-1];
  hl_right_homo = np.cross(hl_homo, [-1, 0, width/2]);
  #hl_right_homo = np.cross(hl_homo, [-1, 0, 5000]);
  hl_right = hl_right_homo[0:2]/hl_right_homo[-1];
  
  return hl_left, hl_right

def compute_horizon(slope_dist, offset_dist, caffe_sz, sz, crop_info, bin_edges):
  
  # setup
  crop_sz, x_inds, y_inds = crop_info

  # get maximum bin
  slope_bin = np.argmax(slope_dist)
  offset_bin = np.argmax(offset_dist)
  print "slope_bin=",slope_bin
  print "offset_bin=",offset_bin
  
  
  # compute (slope, offset)
  slope = bin2val(slope_bin, bin_edges['slope_bin_edges']) 
  offset = bin2val(offset_bin, bin_edges['offset_bin_edges'])
  print "slope=",slope
  print "offset=",offset
  
  
  # (slope, offset) to (left, right)
  offset = offset * caffe_sz[0]
  c = offset / np.cos(np.abs(slope))
  caffe_left = -np.tan(slope)*caffe_sz[1]/2 + c
  caffe_right = np.tan(slope)*caffe_sz[1]/2 + c

  print "caffe_left=",caffe_left
  print "caffe_right=",caffe_right

  # scale back to cropped image
  c_left = caffe_left * (crop_sz[0] / caffe_sz[0])
  c_right = caffe_right * (crop_sz[0] / caffe_sz[0])
  
  print "c_left=",c_left
  print "c_right=",c_right

  # scale back to original image
  center = [(sz[1]+1)/2, (sz[0]+1)/2]
  print "center=",center
  crop_center = [np.dot(x_inds,[.5, .5])-center[0], center[1]-np.dot(y_inds,[.5, .5])]
  print "crop_center",crop_center
  left_tmp = np.asarray([-crop_sz[1]/2, c_left]) + crop_center 
  right_tmp = np.asarray([crop_sz[1]/2, c_right]) + crop_center 
  print "left_tmp=",left_tmp
  print "right_tmp=",right_tmp
  left, right = extrap_horizon(left_tmp, right_tmp, sz[1])
  print "left=",left
  print "right=",right
  SLOPE1 = (right[1]-left[1])/(right[0]-left[0])
  print "SLOPE=",SLOPE1
  OFFSET1 = (right[0]*left[1]-left[0]*right[1])/((np.sqrt((left[0]-right[0])**2+pow(left[1]-right[1],2)))*(1200/224)**2)
  print "OFFSET=",OFFSET1
  
  #print "int(2.9)=",int(2.9)
  return [np.squeeze(left), np.squeeze(right), caffe_left/224, caffe_right/224]


if __name__ == '__main__': 
 
  # image credit:
  # https://commons.wikimedia.org/wiki/File:HFX_Airport_4.jpg
  #fname = 'airport.jpg'
  #fname = '10047212_41697ad358_o.jpg'
  #fname = '10047308_4a06a44dda_o.jpg'
  #fname = '174941563_950bf5e63e_o.jpg'
  #fname = '3049065234_f607dbe8ff_o.jpg'
  #fname = '3151766391_83e43f3d2a_o.jpg'
  #fname = '001000.bmp'
  #fname = '12.jpg'
  #fname = '11.jpg'
  #fname = '000400.jpg'
  #fname = '003338.jpg'
  #fname = '002873.jpg'
  #fname = '002312.jpg'
  #fname = '000011.jpg'
  #fname = '002867.jpg'
  # load bin edges
  fd=open('./Test.txt','r')
  data=[line.strip().split(' ') for line in fd.readlines()]
  datalist=np.array(data)
  datalistImg=datalist[:,0]
  datalistLR=datalist[:,[5,6]]
  print "datalist=", datalist
  print "datalistLR=", datalistLR
  print "datalistImg=", datalistImg
  fd.close()



  bin_edges = sio.loadmat('bins.mat')

  # load network
  #deploy_file = '../models/classification/so_placesvggs_tower/deploy5.net'
  #model_file = '../models/classification/so_places/so_places.caffemodel'
  #model_file = '../models/classification/so_placesvggs/VGG_CNN_S.caffemodel'
  #model_file = '../models/classification/so_placesvggs_tower/snapshots5/solver5_iter_1000good.caffemodel'

  deploy_file = '../models/classification/so_posenet/deploy.net'
  model_file = '../models/classification/so_posenet/so_posenet.caffemodel'
  #deploy_file = '../models/regression/init_places_so_huber/deploy.net'
  #model_file = '../models/regression/init_places_so_huber/init_places_so_huber.caffemodel'
  caffe.set_mode_cpu()
  net = caffe.Net(deploy_file, model_file, caffe.TEST)
  caffe_sz = np.asarray(net.blobs['data'].shape)[2:]
  print "caffe_sz=",caffe_sz

  error=[]
  # preprocess image
  for i in range(200):
      fname='./TestImage/'+datalistImg[i]
      im = caffe.io.load_image(fname)
      sz = im.shape
      center_im, crop_info = center_crop(im)
      caffe_input = preprocess(center_im, caffe_sz)
  #print "im.shape sz=",im.shape
  #print "crop_info:crop_sz, x_inds, y_inds=",crop_info

  # push through the network
      result = net.forward(data=caffe_input, blobs=['prob_slope', 'prob_offset'])
      slope_dist = result['prob_slope'][0]
      offset_dist = result['prob_offset'][0]
  #offset_dist = result['prob_slope'][0]
  #slope_dist = result['prob_offset'][0]
  
  
  #print "slope_dist=",slope_dist
  #print "offset_dist=",offset_dist

  # convert distributions to horizon line 
      left, right, caffe_left, caffe_right = compute_horizon(slope_dist, offset_dist, caffe_sz, sz, crop_info, bin_edges)
  #print left[1], right[1]
  #print 'caffe_left=',caffe_left
  #print 'caffe_light=',caffe_right

      if abs(caffe_left-float(datalistLR[i,0]))>=abs(caffe_right-float(datalistLR[i,1])):
          print 'caffe_left_greater', abs(caffe_left-float(datalistLR[i,0]))
          error.append(float(abs(caffe_left-float(datalistLR[i,0]))))
      else:
          print 'caffe_right_greater', abs(caffe_right-float(datalistLR[i,1]))
          error.append(float(abs(caffe_right-float(datalistLR[i,1]))))

  print "error=",error
  print "index max=",error.index(max(error))
  print "max image=",datalistImg[error.index(max(error))]

  np.array(error)
  average=np.mean(error)
  print "average=", average
  plt.figure(2)
  plt.hist(error,100,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
  plt.title('probability distribution function')
  plt.xlabel('maximum distance from the detection to the ground truth in normalized image')
  plt.ylabel('Number of Pictures')
  plt.show()

  plt.figure(3)
  plt.hist(error, 5000, normed=1, histtype='bar',facecolor='pink',alpha=0.5,cumulative=True)
  plt.title('cumulative distribution function')
  plt.xlabel('maximum distance from the detection to the ground truth in normalized image')
  plt.ylabel('probability')
  plt.show()


  plt.figure(1)
  plt.imshow(im, extent=[-sz[1]/2, sz[1]/2, -sz[0]/2, sz[0]/2])
  plt.plot([left[0], right[0]], [left[1], right[1]], 'r')
  ax = plt.gca();
  ax.autoscale_view('tight')
  #plt.show()



  #if image_name == None:
    #img_path = './data/kernel_feature_picture100/'
  #else:
  img_path = './TestImageResult/'
  #my_path = './data/kernel_feature_picture100/' + image_name + "/"
    #check_file(my_path)
  print 1
  plt.savefig(img_path + fname[12:],dpi=400, bbox_inches="tight")

