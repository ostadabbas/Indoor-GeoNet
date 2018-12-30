from __future__ import division
import os
import math
import scipy.misc
import tensorflow as tf
import numpy as np
from glob import glob
from geonet_model import *
from kitti_eval.pose_evaluation_utils import dump_pose_seq_TUM
from kitti_eval.pose_evaluation_utils import my_dump_pose_seq_TUM
#import pdb

def test_pose(opt):

#    if not os.path.isdir(opt.output_dir):
#        os.makedirs(opt.output_dir)

    ##### init #####
    input_uint8 = tf.placeholder(tf.uint8, [opt.batch_size, 
        opt.img_height, opt.img_width, opt.seq_length * 3], 
        name='raw_input')
    tgt_image = input_uint8[:,:,:,:3]
    src_image_stack = input_uint8[:,:,:,3:]

    model = GeoNetModel(opt, tgt_image, src_image_stack, None)
    fetches = { "pose": model.pred_poses }

    saver = tf.train.Saver([var for var in tf.model_variables()]) 

    ##### load test frames #####
    if opt.pose_test_seq == None:
        seq_names = [direc for direc in os.listdir(opt.dataset_dir) \
                                         if os.path.isdir(os.path.join(opt.dataset_dir,direc))]
    else:
        seq_names = opt.pose_test_seq.split(',')

    #seq_dir = os.path.join(opt.dataset_dir, 'sequences', '%.2d' % opt.pose_test_seq)

    for seq_name in seq_names:

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
            continue

        if not os.path.isdir(os.path.join(opt.output_dir,seq_name,'pose')):
            os.makedirs(os.path.join(opt.output_dir,seq_name,'pose'))
       
        out_dir = os.path.join(opt.output_dir,seq_name,'pose')
                
        N = len(glob(img_dir + '/*.png'))
        #test_frames = ['%.2d %.6d' % (opt.pose_test_seq, n) for n in range(N)]
        #test_frames = ['C1 1 %.10d' %n for n in range(N)]
        #test_frames = ['Traj_54_0_Corridor_0.7 %.10d' %n for n in range(N)]
        if opt.dataset_name == 'kitti_odom':
            test_frames = ['%s %.6d' %(seq_name,n) for n in range(N)]
        else:
            test_frames = ['%s %.10d' %(seq_name,n) for n in range(N)]
    
    ##### load time file #####
    #with open(opt.dataset_dir + 'sequences/%.2d/times.txt' % opt.pose_test_seq, 'r') as f:
    #with open(opt.dataset_dir + '/C1/videos/1/times.txt', 'r') as f:
#    with open(opt.dataset_dir + '/Traj_54_0_Corridor_0.7/times.txt', 'r') as f:
#        times = f.readlines()
#    times = np.array([float(s[:-1]) for s in times])
        times = np.array([i for i in range(N)])
    ##### Go! #####
        max_src_offset = (opt.seq_length - 1) // 2
        with tf.Session() as sess:
            saver.restore(sess, opt.init_ckpt_file)
            for tgt_idx in range(max_src_offset, N-max_src_offset, opt.batch_size):            
                if (tgt_idx-max_src_offset) % 100 == 0:
                    print('Progress: %d/%d' % (tgt_idx-max_src_offset, N))
    
                inputs = np.zeros((opt.batch_size, opt.img_height,
                         opt.img_width, 3*opt.seq_length), dtype=np.uint8)
    
                for b in range(opt.batch_size):
                    idx = tgt_idx + b
                    if idx >= N-max_src_offset:
                        break
                    image_seq = load_image_sequence(opt.dataset_dir,
                                                    test_frames,
                                                    idx,
                                                    opt.seq_length,
                                                    opt.img_height,
                                                    opt.img_width,
                                                    opt.dataset_name)
                    inputs[b] = image_seq
    
                pred = sess.run(fetches, feed_dict={input_uint8: inputs})
                pred_poses = pred['pose']
                # Insert the target pose [0, 0, 0, 0, 0, 0] 
                pred_poses = np.insert(pred_poses, max_src_offset, np.zeros((1,6)), axis=1)
    
                for b in range(opt.batch_size):
                    idx = tgt_idx + b
                    if idx >=N-max_src_offset:
                        break
                    pred_pose = pred_poses[b]                
                    curr_times = times[idx - max_src_offset:idx + max_src_offset + 1]
                    #out_file = opt.output_dir + '%.6d.txt' % (idx - max_src_offset)
                    out_file = out_dir + '/%.10d.txt' % (idx - max_src_offset)
                    #dump_pose_seq_TUM(out_file, pred_pose, curr_times)
                    my_dump_pose_seq_TUM(out_file, pred_pose, curr_times)

def load_image_sequence(dataset_dir, 
                        frames, 
                        tgt_idx, 
                        seq_length, 
                        img_height, 
                        img_width,
                        dataset_name):
    half_offset = int((seq_length - 1)/2)
    for o in range(-half_offset, half_offset+1):
        curr_idx = tgt_idx + o
        #curr_drive, curr_frame_id = frames[curr_idx].split(' ')
        #curr_drive, curr_path, curr_frame_id = frames[curr_idx].split(' ')
        curr_path, curr_frame_id = frames[curr_idx].split(' ')
        #img_file = os.path.join(
        #    dataset_dir, 'sequences', '%s/image_2/%s.png' % (curr_drive, curr_frame_id))
        #img_file = os.path.join(
                #dataset_dir, '%s/videos/%s/my_frames/%s.png' % (curr_drive, curr_path, curr_frame_id))
        #pdb.set_trace()
        if dataset_name == 'kitti_odom':
            img_file = os.path.join(
                    dataset_dir, '%s/image_2/%s.png' % (curr_path, curr_frame_id))
        elif dataset_name == 'other':
            img_file = os.path.join(
                    dataset_dir, '%s/%s.png' % (curr_path, curr_frame_id))
        elif dataset_name == 'kitti_raw':
            img_file = os.path.join(
                    dataset_dir, '%s/image_02/data/%s.png' % (curr_path, curr_frame_id))
        elif dataset_name == 'rms_hallway':
            img_file = os.path.join(
                    dataset_dir, '%s/my_frames/%s.png' % (curr_path, curr_frame_id))
        elif dataset_name == 'MobileRGBD':
            img_file = os.path.join(
                    dataset_dir, '%s/video/%s.png' % (curr_path, curr_frame_id))
                    
        curr_img = scipy.misc.imread(img_file)
        curr_img = scipy.misc.imresize(curr_img, (img_height, img_width))
        if o == -half_offset:
            image_seq = curr_img
        elif o == 0:
            image_seq = np.dstack((curr_img, image_seq))
        else:
            image_seq = np.dstack((image_seq, curr_img))
    return image_seq
