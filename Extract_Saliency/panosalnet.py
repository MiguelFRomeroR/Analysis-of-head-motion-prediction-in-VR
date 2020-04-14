# Extract saliency using PanoSalNet from the videos
# input: model config file: panosalnet_test.prototxt
#       model weight: 'panosalnet_iter_800.caffemodel
#       input image: test.png
# output: salient map

import cv2
import numpy as np
import caffe
import timeit
import os
import pickle
import argparse

#usage python panosalnet.py -gpu_id 3 -dataset_name David_MMSys_18

parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')
parser.add_argument('-gpu_id', action='store', dest='gpu_id', help='The gpu used to train this network.')
parser.add_argument('-dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')

args = parser.parse_args()

GPU_ID = args.gpu_id
dataset_name = args.dataset_name

assert dataset_name in ['AVTrack360', 'PAMI_18', 'CVPR_18', 'NOSSDAV_17', 'David_MMSys_18']

caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

FILE_MODEL_CONFIG = 'panosalnet_test.prototxt'
FILE_MODEL_WEIGHT = 'panosalnet_iter_800.caffemodel'

if dataset_name == 'AVTrack360':
    VIDEOS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
    FOLDER_IMAGES = './AVTrack360/dataset/videos/cut_wogrey'
    FOLDER_SALIENCY = './AVTrack360/extract_saliency/saliency'
elif dataset_name == 'PAMI_18':
    FOLDER_IMAGES = './Xu_PAMI_18/dataset/videos'
    FOLDER_SALIENCY = './Xu_PAMI_18/extract_saliency/saliency'
elif dataset_name == 'CVPR_18':
    FOLDER_IMAGES = './Xu_CVPR_18/dataset/Videos'
    FOLDER_SALIENCY = './Xu_CVPR_18/extract_saliency/saliency'
elif dataset_name == 'NOSSDAV_17':
    print ("Use function create_saliency_maps from Fan_NOSSDAV_17/Reading_Dataset")
elif dataset_name == 'David_MMSys_18':
    FOLDER_IMAGES = './David_MMSys_18/dataset/Videos/Stimuli'
    FOLDER_SALIENCY = './David_MMSys_18/extract_saliency/saliency'

def post_filter(_img):
    result = np.copy(_img)
    result[:3, :3] = _img.min()
    result[:3, -3:] = _img.min()
    result[-3:, :3] = _img.min()
    result[-3:, -3:] = _img.min()
    return result


net = caffe.Net(FILE_MODEL_CONFIG, caffe.TEST)
net.copy_from(FILE_MODEL_WEIGHT)

C, H, W = net.blobs['data1'].data[0].shape

VIDEOS = [d for d in os.listdir(FOLDER_IMAGES) if os.path.isdir(os.path.join(FOLDER_IMAGES, d))]

for VIDEO in VIDEOS:
    sal_per_vid = {}
    VIDEO_SAL_FOLDER = ('%s/%s') % (FOLDER_SALIENCY, VIDEO)
    if not os.path.exists(VIDEO_SAL_FOLDER):
        os.makedirs(VIDEO_SAL_FOLDER)
    VIDEO_FOLDER = ('%s/%s') % (FOLDER_IMAGES, VIDEO)
    for IMAGE_NAME in os.listdir(VIDEO_FOLDER):
        FILE_DIR = ("%s/%s") % (VIDEO_FOLDER, IMAGE_NAME)
        img = cv2.imread(FILE_DIR)
        mu = [104.00698793, 116.66876762, 122.67891434]
        input_scale = 0.0078431372549

        dat = cv2.resize(img, (W, H))
        dat = dat.astype(np.float32)
        dat -= np.array(mu)  # mean values
        dat *= input_scale
        dat = dat.transpose((2, 0, 1))

        net.blobs['data1'].data[0] = dat
        net.forward(start='conv1')

        salient = net.blobs['deconv1'].data[0][0]
        salient = (salient * 1.0 - salient.min())
        salient = (salient / salient.max()) * 255
        salient = post_filter(salient)

        import scipy.misc

        OUTPUT_FILE = ('%s/%s') % (VIDEO_SAL_FOLDER, IMAGE_NAME)
        scipy.misc.imsave(OUTPUT_FILE, salient)
        frame_id = IMAGE_NAME.split('.')[0].split('_')[-1]
        sal_per_vid[frame_id] = salient
        print('saved image %s' % (OUTPUT_FILE))
    pickle.dump(sal_per_vid, open(('%s/%s') % (VIDEO_SAL_FOLDER, VIDEO), 'wb'))
