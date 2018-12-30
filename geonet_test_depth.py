from __future__ import division
import tensorflow as tf
import numpy as np
import os
import PIL.Image as pil
from geonet_model import *
from glob import glob

def test_depth(opt):
    ##### load testing list #####
    #with open('data/kitti/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
#    with open('data/%s/test_files_%s.txt' % (opt.dataset_name, opt.depth_test_split), 'r') as f:
#    #with open('data/MobileRGBD/test_files_%s.txt' % opt.depth_test_split, 'r') as f:
#        test_files = f.readlines()
#        test_files = [opt.dataset_dir + t[:-1] for t in test_files]
#    if not os.path.exists(opt.output_dir):
#        os.makedirs(opt.output_dir)
    
    if opt.depth_test_seq == None:
        seq_names = [direc for direc in os.listdir(opt.dataset_dir) \
                                         if os.path.isdir(os.path.join(opt.dataset_dir,direc))]
    else:
        seq_names = opt.depth_test_seq.split(',')
    
    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size,
                opt.img_height, opt.img_width, 3], name='raw_input')

    model = GeoNetModel(opt, input_uint8, None, None)
    fetches = { "depth": model.pred_depth[0] }

    saver = tf.train.Saver([var for var in tf.model_variables()])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
      
    ##### Go #####
    with tf.Session(config=config) as sess:
        saver.restore(sess, opt.init_ckpt_file)
        for seq_name in seq_names:
            test_files = test_file_generator(opt, seq_name)  
            pred_all = []
            for t in range(0, len(test_files), opt.batch_size):
                
                if t % 100 == 0:
                    print('processing: %d/%d' % (t, len(test_files)))
                inputs = np.zeros(
                    (opt.batch_size, opt.img_height, opt.img_width, 3),
                    dtype=np.uint8)
    
                for b in range(opt.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    #fh = open(test_files[idx], 'r')
                    fh = test_files[idx]
                    raw_im = pil.open(fh)
                    scaled_im = raw_im.resize((opt.img_width, opt.img_height), pil.ANTIALIAS)
                    inputs[b] = np.array(scaled_im)
    
                pred = sess.run(fetches, feed_dict={input_uint8: inputs})
                for b in range(opt.batch_size):
                    idx = t + b
                    if idx >= len(test_files):
                        break
                    pred_all.append(pred['depth'][b,:,:,0])
    
            np.save(opt.output_dir + '/%s/depth/' % seq_name + os.path.basename(opt.init_ckpt_file), pred_all)


def test_file_generator(opt, seq_name):

    seq_dir = os.path.join(opt.dataset_dir, '%s' % seq_name)
    if opt.dataset_name == 'kitti_odom':
        img_dir = os.path.join(seq_dir, 'image_2')
    elif opt.dataset_name == 'other':
        img_dir = seq_dir
    elif opt.dataset_name == 'kitti_raw':
        img_dir = os.path.join(seq_dir, 'image_02','data')
    elif opt.dataset_name == 'rms_hallway' and seq_name != 'extra':
        #img_dir = os.path.join(seq_dir, 'C1', 'videos', '1', 'my_frames')
        img_dir = os.path.join(seq_dir, 'my_frames')
    elif opt.dataset_name == 'MobileRGBD' and seq_name != 'extra':
        #img_dir = os.path.join(seq_dir, 'Traj_54_0_Corridor_0.7', 'video')
        img_dir = os.path.join(seq_dir, 'video')
    else:
        test_frames = []
        return test_frames

    if not os.path.isdir(os.path.join(opt.output_dir,seq_name,'depth')):
        os.makedirs(os.path.join(opt.output_dir,seq_name,'depth'))
            
    N = len(glob(img_dir + '/*.png'))
    #test_frames = ['%.2d %.6d' % (opt.pose_test_seq, n) for n in range(N)]
    #test_frames = ['C1 1 %.10d' %n for n in range(N)]
    #test_frames = ['Traj_54_0_Corridor_0.7 %.10d' %n for n in range(N)]
    if opt.dataset_name == 'kitti_odom':
        test_frames = ['%s/%.6d.png' %(img_dir,n) for n in range(N)]
    else:
        test_frames = ['%s/%.10d.png' %(img_dir,n) for n in range(N)]
                
    return test_frames
    