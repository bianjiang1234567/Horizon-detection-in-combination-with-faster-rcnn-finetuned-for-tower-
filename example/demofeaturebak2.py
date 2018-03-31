#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from skimage.viewer import ImageViewer
#CLASSES = ('__background__',
#           'tower')

CLASSES = ('__background__',
           'tower')
           #'aeroplane', 'bicycle', 'bird', 'boat',
           #'bottle', 'bus', 'car', 'cat', 'chair',
           #'cow', 'diningtable', 'dog', 'horse',
          # 'motorbike', 'person', 'pottedplant',
          # 'sheep', 'sofa', 'train', 'tvmonitor')


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  #'ZF_faster_rcnn_finaltower4848.caffemodel')}
                  #'ZF_faster_rcnn_finaltower100.caffemodel')}
                  'ZF_faster_rcnn_final.caffemodel')}
                  #'ZF.v2.caffemodel')}
                  #'ZF_faster_rcnn_finalbak.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def save_feature_picture(data, name, image_name=None, padsize=1, padval=1):
    print "data.shape: ", data.shape
    data = data[0]
    #print "data.shape: ", data.shape

    #data = data[1]
    #print "data.shape2: ", data.shape

    #data = data[2]
    #print "data.shape2: ", data.shape

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    #print "padding: ", padding
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    #print "data.shape2: ", data.shape

    #np.reshape()
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    #print "data.shape3: ", data.shape, n
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    #print "data.shape4: ", data.shape
    plt.figure()
    plt.imshow(data, cmap='gray')
    plt.axis('off')
    # plt.show()
    
    #if image_name == None:
    #    img_path = './data/feature_picture/'
    #else:
    #    img_path = './data/feature_picture100/' + image_name + "/"
    #    my_path='./data/feature_picture100/' + image_name + "/"
    #    check_file(my_path)
    #plt.savefig(img_path + name + ".jpg", dpi=400, bbox_inches="tight")

def check_file(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_kernel_picture(data,name,image_name):
    n=int(np.ceil(np.sqrt(data.shape[0])))
    padding=((0,n**2-data.shape[0]),(0,0),(0,0)) + ((0, 0),) * (data.ndim - 3)
    #print data.ndim , padding

    data = np.pad(data, padding, mode='constant', constant_values=(0.5, 0.5))

    #data = data.reshape(10, 10, 3, 7, 7).transpose(2,0,1,3,4)


    #data = data.reshape(3, 7*n, 7*n)

    #data = data.reshape(96, 3, 7, 7).transpose((0, 2, 3, 1))

    print "kernel padding shape", data.shape

    #type(data)
    #print data[1].shape
    #data[0]=data[0].reshape(n,n)

    datafinal=data[0]
    #print "datafinal.shape", datafinal.shape


    #print datafinal.min(), datafinal.max()

    #plt.autoscale(datafinal)
    #viewer = ImageViewer(datafinal)
    #viewer.show()

    #plt.figure()
    #plt.imshow(datafinal[:,:,0],cmap="jet")
    #plt.axis('off')
    
    #datainterval = np.ones(shape=(79,1,3))color
    datainterval = np.ones(shape=(79,1))
    #datainterval2 = np.ones(shape=(1, 7, 3))color
    datainterval2 = np.ones(shape=(1, 7))
    #datainterval3 = np.ones(shape=(7, 7, 3))color
    datainterval3 = np.ones(shape=(7, 7))
    for i in range(100):
        datar = data[i, 0, :, :]
        #print datar
        minr =datar.min()
        maxr =datar.max()
        #print minr-maxr
        if minr!=maxr:
            datar = abs(1/(maxr-minr)*datar+minr/(minr-maxr))
        #else:
            #continue
            #datar = abs(datar/maxr)
        #print datar
        #print datar.shape
        datag = data[i, 1, :, :]
        ming = datag.min()
        maxg = datag.max()
        # print minr-maxr
        if ming != maxg:
            datag = abs(1 / (maxg - ming) * datag+ ming / (ming - maxg))
        #else:
            #continue
            #datag = abs(datag/maxg)
        datab = data[i, 2, :, :]
        minb = datab.min()
        maxb = datab.max()
        # print minr-maxr
        if minb != maxb:
            datab = abs(1 / (maxb - minb) * datab + minb / (minb - maxb))
        #else:
            #continue
            #datab = abs(datab/maxb)
        #datafinal=np.array([datar,datag,datab]) color
        #datafinal1=datafinal.reshape(7, 7, 3) color
        
        datafinal1=datar
        #print "datar.shape", datar.shape
        
        
        
        if i==0:
            datafinal2=datafinal1
        if 0<i<10:
            datafinal2=np.row_stack((datafinal2, datainterval2, datafinal1))

        if i == 10:
            datafinal3 = datafinal1
        if 11<=i<20:
            datafinal3 = np.row_stack((datafinal3, datainterval2, datafinal1))

        if i == 20:
            datafinal4 = datafinal1
        if 21 <= i < 30:
            datafinal4 = np.row_stack((datafinal4, datainterval2, datafinal1))

        if i == 30:
            datafinal5 = datafinal1
        if 31 <= i < 40:
            datafinal5 = np.row_stack((datafinal5, datainterval2, datafinal1))

        if i == 40:
            datafinal6 = datafinal1
        if 41 <= i < 50:
            datafinal6 = np.row_stack((datafinal6, datainterval2, datafinal1))

        if i == 50:
            datafinal7 = datafinal1
        if 51 <= i < 60:
            datafinal7 = np.row_stack((datafinal7, datainterval2, datafinal1))

        if i == 60:
            datafinal8 = datafinal1
        if 61 <= i < 70:
            datafinal8 = np.row_stack((datafinal8, datainterval2, datafinal1))

        if i == 70:
            datafinal9 = datafinal1
        if 71 <= i < 80:
            datafinal9 = np.row_stack((datafinal9, datainterval2, datafinal1))

        if i == 80:
            datafinal10 = datafinal1
        if 81 <= i < 90:
            datafinal10 = np.row_stack((datafinal10, datainterval2, datafinal1))

        if i == 90:
            datafinal11 = datafinal1
        if 91 <= i < 96:
            datafinal11 = np.row_stack((datafinal11, datainterval2, datafinal1))
        if 96 <= i < 100:
            datafinal11 = np.row_stack((datafinal11, datainterval2, datainterval3))


        #elif i / 10 == 1:
        #    datafinal4 = np.row_stack((datafinal2, datafinal1))
        #datafinal2=np.row_stack((datafinal2,datafinal1))
        #print datafinal1.shape

    datafinal = np.hstack((datafinal2, datainterval, datafinal3, datainterval,datafinal4, datainterval,datafinal5, datainterval,datafinal6,datainterval,datafinal7,
    datainterval, datafinal8,datainterval,datafinal9,datainterval,datafinal10, datainterval,datafinal11))
    #datafinal = np.hstack((datafinal2, datainterval, datafinal3))
    #datafinal = np.hstack((datafinal, datafinal2))
    #datafinal = np.column_stack((datafinal, datafinal5))
    #datafinal1000 = np.column_stack((datafinal100, datafinal101))

    plt.figure()
    plt.imshow(datafinal)
    plt.figure()
    plt.imshow(datafinal, cmap='gray')

    #plt.figure()
    #plt.matshow(data[0, :, :])
    #plt.figure()
    #plt.imshow(data[2, :, :], cmap='winter')
    #plt.axis('off')
    #plt.figure()
    #plt.imshow(data[2,:,:], cmap='gray')

    plt.axis('off')

    if image_name == None:
        img_path = './data/kernel_feature_picture100/'
    else:
        img_path = './data/kernel_feature_picture100/' + image_name + "/"
        my_path = './data/kernel_feature_picture100/' + image_name + "/"
        check_file(my_path)

        plt.savefig(img_path + name + ".jpg", dpi=400, bbox_inches="tight")


    #plt.matshow(data[0,:,:])

    #plt.matshow(data[1,:,:])

    #plt.matshow(data[2,:,:])
    #plt.axis('off')


    #datafinal=np.zeros()
    #print datafinal

    #for j in range(3):
    #    for i in range(100):
           #datafinal[i/10,i%10]=data[i, j, :, :]
           #np.reshape()
           #help(data[:, j, :, :])
           #plt.figure()
           #plt.imshow(datafinal, cmap='gray')
           #plt.axis('off')


def save_kerne2_picture(data,name,image_name):
    n=int(np.ceil(np.sqrt(data.shape[0])))
    padding=((0,n**2-data.shape[0]),(0,0),(0,0)) + ((0, 0),) * (data.ndim - 3)
    #print data.ndim , padding

    data = np.pad(data, padding, mode='constant', constant_values=(0.5, 0.5))

    #data = data.reshape(10, 10, 3, 7, 7).transpose(2,0,1,3,4)


    #data = data.reshape(3, 7*n, 7*n)

    #data = data.reshape(96, 3, 7, 7).transpose((0, 2, 3, 1))

    print "kernel2 padding shape", data.shape

    #type(data)
    #print data[1].shape
    #data[0]=data[0].reshape(n,n)

    datafinal=data[0]
    #print "datafinal.shape", datafinal.shape


    #print datafinal.min(), datafinal.max()

    #plt.autoscale(datafinal)
    #viewer = ImageViewer(datafinal)
    #viewer.show()

    #plt.figure()
    #plt.imshow(datafinal[:,:,0],cmap="jet")
    #plt.axis('off')
    
    #datainterval = np.ones(shape=(79,1,3))color
    datainterval = np.ones(shape=(95,1))
    #datainterval2 = np.ones(shape=(1, 7, 3))color
    datainterval2 = np.ones(shape=(1, 5))
    #datainterval3 = np.ones(shape=(7, 7, 3))color
    datainterval3 = np.ones(shape=(7, 5))
    for i in range(256):
        datar = data[i, 0, :, :]
        #print datar
        minr =datar.min()
        maxr =datar.max()
        #print minr-maxr
        if minr!=maxr:
            datar = abs(1/(maxr-minr)*datar+minr/(minr-maxr))
        #else:
            #continue
            #datar = abs(datar/maxr)
        #print datar
        #print datar.shape
        datag = data[i, 1, :, :]
        ming = datag.min()
        maxg = datag.max()
        # print minr-maxr
        if ming != maxg:
            datag = abs(1 / (maxg - ming) * datag+ ming / (ming - maxg))
        #else:
            #continue
            #datag = abs(datag/maxg)
        datab = data[i, 2, :, :]
        minb = datab.min()
        maxb = datab.max()
        # print minr-maxr
        if minb != maxb:
            datab = abs(1 / (maxb - minb) * datab + minb / (minb - maxb))
        #else:
            #continue
            #datab = abs(datab/maxb)
        #datafinal=np.array([datar,datag,datab]) color
        #datafinal1=datafinal.reshape(7, 7, 3) color
        
        datafinal1=datar
        #print "datar.shape", datar.shape
        
        
        
        if i==0:
            datafinal2=datafinal1
        if 1<=i<=15:
            datafinal2=np.row_stack((datafinal2, datainterval2, datafinal1))

        if i == 16:
            datafinal3 = datafinal1
        if 17<=i<32:
            datafinal3 = np.row_stack((datafinal3, datainterval2, datafinal1))

        if i == 32:
            datafinal4 = datafinal1
        if 33 <= i < 48:
            datafinal4 = np.row_stack((datafinal4, datainterval2, datafinal1))

        if i == 48:
            datafinal5 = datafinal1
        if 49 <= i < 64:
            datafinal5 = np.row_stack((datafinal5, datainterval2, datafinal1))

        if i == 64:
            datafinal6 = datafinal1
        if 65 <= i < 80:
            datafinal6 = np.row_stack((datafinal6, datainterval2, datafinal1))

        if i == 80:
            datafinal7 = datafinal1
        if 81 <= i < 96:
            datafinal7 = np.row_stack((datafinal7, datainterval2, datafinal1))

        if i == 96:
            datafinal8 = datafinal1
        if 97 <= i < 112:
            datafinal8 = np.row_stack((datafinal8, datainterval2, datafinal1))

        if i == 112:
            datafinal9 = datafinal1
        if 113 <= i < 128:
            datafinal9 = np.row_stack((datafinal9, datainterval2, datafinal1))

        if i == 128:
            datafinal10 = datafinal1
        if 129 <= i < 144:
            datafinal10 = np.row_stack((datafinal10, datainterval2, datafinal1))

        if i == 144:
            datafinal11 = datafinal1
        if 145 <= i < 160:
            datafinal11 = np.row_stack((datafinal11, datainterval2, datafinal1))

            
        if i == 160:
            datafinal12 = datafinal1
        if 161 <= i < 176:
            datafinal12 = np.row_stack((datafinal12, datainterval2, datafinal1))

        
        if i == 176:
            datafinal13 = datafinal1
        if 177 <= i < 192:
            datafinal13 = np.row_stack((datafinal13, datainterval2, datafinal1))
            
        
        if i == 192:
            datafinal14 = datafinal1
        if 193 <= i < 208:
            datafinal14 = np.row_stack((datafinal14, datainterval2, datafinal1))
            
        
        if i == 208:
            datafinal15 = datafinal1
        if 209 <= i < 224:
            datafinal15 = np.row_stack((datafinal15, datainterval2, datafinal1))
            
        
        if i == 224:
            datafinal16 = datafinal1
        if 225 <= i < 240:
            datafinal16 = np.row_stack((datafinal16, datainterval2, datafinal1))
      
      
        if i == 240:
            datafinal17 = datafinal1
        if 241 <= i < 256:
            datafinal17 = np.row_stack((datafinal17, datainterval2, datafinal1))


        #elif i / 10 == 1:
        #    datafinal4 = np.row_stack((datafinal2, datafinal1))
        #datafinal2=np.row_stack((datafinal2,datafinal1))
        #print datafinal1.shape
    print "datafinal2.shape", datafinal2.shape
    print "datafinal3.shape", datafinal3.shape
    print "datafinal4.shape", datafinal4.shape
    print "datafinal5.shape", datafinal5.shape
    print "datafinal6.shape", datafinal6.shape
    print "datafinal7.shape", datafinal7.shape
    print "datafinal8.shape", datafinal8.shape
    print "datafinal9.shape", datafinal9.shape
    print "datafinal10.shape", datafinal10.shape
    print "datafinal11.shape", datafinal11.shape
    print "datafinal12.shape", datafinal12.shape
    print "datafinal13.shape", datafinal13.shape
    print "datafinal14.shape", datafinal14.shape
    print "datafinal15.shape", datafinal15.shape
    print "datafinal16.shape", datafinal16.shape
    print "datafinal17.shape", datafinal17.shape
    datafinal = np.hstack((datafinal2, datainterval, datafinal3, datainterval,datafinal4, datainterval,datafinal5, datainterval,datafinal6,datainterval,datafinal7,
    datainterval, datafinal8,datainterval,datafinal9,datainterval,datafinal10, datainterval,datafinal11, datainterval,datafinal12, datainterval,datafinal13, datainterval,datafinal14, datainterval,datafinal15, datainterval     
    ,datafinal16, datainterval,datafinal17))
    #datafinal = np.hstack((datafinal2, datainterval, datafinal3))
    #datafinal = np.hstack((datafinal, datafinal2))
    #datafinal = np.column_stack((datafinal, datafinal5))
    #datafinal1000 = np.column_stack((datafinal100, datafinal101))

    plt.figure()
    plt.imshow(datafinal)
    plt.figure()
    plt.imshow(datafinal, cmap='gray')

    #plt.figure()
    #plt.matshow(data[0, :, :])
    #plt.figure()
    #plt.imshow(data[2, :, :], cmap='winter')
    #plt.axis('off')
    #plt.figure()
    #plt.imshow(data[2,:,:], cmap='gray')

    plt.axis('off')

    if image_name == None:
        img_path = './data/kernel_feature_picture100/'
    else:
        img_path = './data/kernel_feature_picture100/' + image_name + "/"
        my_path = './data/kernel_feature_picture100/' + image_name + "/"
        check_file(my_path)

        plt.savefig(img_path + name + ".jpg", dpi=400, bbox_inches="tight")



def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()

    #print "net.params", net.params
    print "net.params", net.params

    for k, v in net.params.items():
        #if k.find("conv1") > -1:
        if k=="conv1":
            print k,v
            #print v.blobs_
            print "conv1 kernel shape", v[0].data.shape
            #plt.figure()
            #plt.imshow(v[0].data[2,2,:,:], cmap='gray')
            #plt.axis('off')


            #print dir(v[0])
            save_kernel_picture(v[0].data, k.replace("/", ""), image_name)

    for k, v in net.params.items():
        #if k.find("conv1") > -1:
        if k=="conv2":
            #print k,v
            #print v.blobs_
            print "conv2 kernel shape", v[0].data.shape
            #plt.figure()
            #plt.imshow(v[0].data[2,2,:,:], cmap='gray')
            #plt.axis('off')


            #print dir(v[0])
            save_kerne2_picture(v[0].data, k.replace("/", ""), image_name)



    #print ("net.blobs.items():",net.blobs.items())
    for k, v in net.blobs.items():
        if k.find("conv")>-1 or k.find("pool")>-1 or k.find("rpn")>-1:
            #k.replace("/", "")
            #print v, k, type(v)

            #print k,v

            save_feature_picture(v.data, k.replace("/", ""), image_name)#net.blobs["conv1_1"].data, "conv1_1")
            #save_feature_picture(v.params, k.replace("/", ""), image_name)  # net.blobs["conv1_1"].data, "conv1_1")

    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')# 'faster_rcnn_testimagenet.pt'
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    #im_names = ['000001.jpg','000003.jpg',
    #           '000008.jpg', '000008.jpg',
    #            '000012.jpg', '000022.jpg']

    #im_names = ['000456.jpg','000542.jpg','001150.jpg','001763.jpg','004545.jpg']

    im_names = ['000100.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
