# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 21:40:05 2018

@author: Amir
"""

from my_utils import *
import cv2
from glob import glob
from matplotlib import pyplot as plt
import argparse
import os
from skimage.measure import compare_ssim as ssim


def load_camera_matrix(files_dir, frame_id, line_id):

    cam_file = os.path.join(files_dir, '%.10d.txt' %frame_id)

    filedata = read_camera_matrix_file(cam_file)
    R = np.reshape(filedata['R%s' %line_id], (3, 4))
    cam_mat = np.vstack([R,np.array([[0,0,0,1]])])
    return cam_mat


def read_camera_matrix_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a camera matrix file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                    data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                    pass
    return data


def load_intrinsics(dataset_dir, dataset_name, seq_name):
    
    if dataset_name == 'kitti_raw':
        calib_file = os.path.join(dataset_dir, 'calib_cam_to_cam.txt')
        filedata = read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect' + '_02'], (3, 4))
        intrinsics = P_rect[:3, :3]
    elif dataset_name == 'kitti_odom':
        calib_file = os.path.join(dataset_dir, '%s/calib.txt' % seq_name)
        proj_c2p, _ = read_calib_file(calib_file)
        intrinsics = proj_c2p[:3, :3]
    elif dataset_name in ['rms_hallway','MobileRGBD','other']:
        calib_file = os.path.join(dataset_dir, 'calib_cam.txt')
        filedata = read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect'], (3, 4))
        intrinsics = P_rect[:3, :3]

    return intrinsics

def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                    data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                    pass
    return data

def read_calib_file(filepath, cid=2):
    """Read in a calibration file and parse into a dictionary."""
    with open(filepath, 'r') as f:
        C = f.readlines()
    def parseLine(L, shape):
        data = L.split()
        data = np.array(data[1:]).reshape(shape).astype(np.float32)
        return data
    proj_c2p = parseLine(C[cid], shape=(3,4))
    proj_v2c = parseLine(C[-1], shape=(3,4))
    filler = np.array([0, 0, 0, 1]).reshape((1,4))
    proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
    return proj_c2p, proj_v2c

def scale_intrinsics(mat, sx, sy):
    out = np.copy(mat)
    out[0,0] *= sx
    out[0,2] *= sx
    out[1,1] *= sy
    out[1,2] *= sy
    return out


dataset_dir_list=['S:/Users/Amir/GeoNet/MobileRGBD/Corridor/'] #edit this line
                  
model_name_list=['model_sn', 'model-145000', 'model-30000'] #edit this line
eval_seq_list=['Traj_54_0_Corridor_1.1'] #edit this line
predictions_dir_list=['S:/Users/Amir/GeoNet/MobileRGBD/'] #edit this line

dataset_name_list=['MobileRGBD'] #edit this line

net_name=['predictions_geonet_kitti/','predictions_indoor_geonet_RSM/','predictions_weakly_supervised/'] #edit this line

s_rate = [1]

tgt_image_stack=[]
src_image_stack=[]
src_image_rec_stack={'first':[],'second':[],'third':[]}
depth_stack={'first':[],'second':[],'third':[]}
m_name = ['first','second', 'third']
scale = [6,1,1]

for ii in range(1):
    
    dataset_dir = dataset_dir_list[ii]
    dataset_name = dataset_name_list[ii]
    eval_seq = eval_seq_list[ii]
    predictions_dir_root = predictions_dir_list[ii]
    
    for jj in range(3):
        
        predictions_dir = predictions_dir_root + net_name[jj]
        model_name = model_name_list[jj]
        
        if eval_seq == None:
            seq_names = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,f))]
        else:
            seq_names = eval_seq.split(',')
        
        
        for seq_name in seq_names:
            
            seq_dir = os.path.join(dataset_dir, '%s' % seq_name)
            if dataset_name == 'kitti_odom':
                img_dir = os.path.join(seq_dir, 'image_2')
            elif dataset_name == 'other':
                img_dir = seq_dir
            elif dataset_name == 'kitti_raw':
                img_dir = os.path.join(seq_dir, 'image_02','data')
            elif dataset_name == 'rms_hallway' and seq_name != 'extra':
                #img_dir = os.path.join(seq_dir, 'C1', 'videos', '1', 'my_frames')
                img_dir = os.path.join(seq_dir, 'my_frames')
            elif dataset_name == 'MobileRGBD' and seq_name != 'extra':
                #img_dir = os.path.join(seq_dir, 'Traj_54_0_Corridor_0.7', 'video')
                img_dir = os.path.join(seq_dir, 'video')
            else:
                continue
        
            depth = np.load(os.path.join(predictions_dir, seq_name, 'depth') + '/%s.npy' % model_name)
            N, height, width = depth.shape
            depth = depth.reshape((N, 1, height, width))
            
            
            intrinsics = load_intrinsics(dataset_dir, dataset_name, seq_name)
        #    intrinsics = np.array([[256, 0, 128],
        #                           [0, 256, 72],
        #                           [0, 0, 1]]).astype('float32')
            
            if dataset_name == 'kitti_odom':
                img = cv2.imread(img_dir + '/%.6d.png' % 0)
            else:
                img = cv2.imread(img_dir + '/%.10d.png' % 0)
                
            intrinsics = scale_intrinsics(intrinsics, width/img.shape[1] , height/img.shape[0])
            intrinsics = intrinsics.reshape((1,3,3))
            #intrinsics_tensor = tf.convert_to_tensor(intrinsics)
            err = 0
            err_ssim = 0
            files_dir = os.path.join(predictions_dir, seq_name, 'pose')
            data_dir = img_dir
            N =len(glob(files_dir+'/*.txt'))
            for n in range(N):
                       
                       if n % 1 != 0:
                           continue
                       
                       cam_mat = load_camera_matrix(files_dir, n, n+1)
                       cam_mat = cam_mat.reshape((1,4,4)).astype('float32')
                       #cam_mat_tensor = tf.convert_to_tensor(cam_mat)
                       #depth_tensor = tf.convert_to_tensor(depth[n])
                       rigid_flow, src_pixel_coords, tgt_pixel_coords = my_compute_rigid_flow(depth[n]*scale[jj], cam_mat, intrinsics, reverse_pose=True)
            #           src_pixel_coords = tf.Session().run(src_pixel_coords).reshape((-1,2))
            #           tgt_pixel_coords = tf.Session().run(tgt_pixel_coords).reshape((-1,2))
                       src_pixel_coords = src_pixel_coords.reshape((-1,2))
                       tgt_pixel_coords = tgt_pixel_coords.reshape((-1,2))
                       src_pixel_coords = np.floor(src_pixel_coords).astype('int') + 1
                       tgt_pixel_coords = np.floor(tgt_pixel_coords).astype('int') + 1
                       
                       if dataset_name == 'kitti_odom':
                            tgt_image = cv2.imread(data_dir + '/%.6d.png' %n)
                            src_image = cv2.imread(data_dir + '/%.6d.png' %(n+1))
                       else:
                             tgt_image = cv2.imread(data_dir + '/%.10d.png' %n)
                             src_image = cv2.imread(data_dir + '/%.10d.png' %(n+1))
                             
                       tgt_image = cv2.resize(tgt_image, (width, height)).astype('float')
                       src_image = cv2.resize(src_image, (width, height)).astype('float')
                       src_image_rec = np.zeros((height, width, 3))
                       
                       cnt = 0
                       err_img = 0
                       for i in range(height*width):
                           if src_pixel_coords[i,0]<width and src_pixel_coords[i,1]<height and src_pixel_coords[i,0]>=0 and src_pixel_coords[i,1]>=0:
                               if tgt_pixel_coords[i,0]<width and tgt_pixel_coords[i,1]<height and tgt_pixel_coords[i,0]>=0 and tgt_pixel_coords[i,1]>=0:
                                   src_image_rec[src_pixel_coords[i,1], src_pixel_coords[i,0]] = tgt_image[tgt_pixel_coords[i,1], tgt_pixel_coords[i,0]]
                                   err_img = err_img + np.sum((src_image_rec[src_pixel_coords[i,1], src_pixel_coords[i,0]] - src_image[src_pixel_coords[i,1], src_pixel_coords[i,0]])**2)
                                   cnt = cnt + 1
                       err = err + np.sqrt(err_img/(cnt*3))
                       err_ssim = err_ssim + ssim(src_image, src_image_rec, data_range=src_image_rec.max()-src_image_rec.min(), multichannel=True)
                       
                       if n % s_rate[ii] == 0:
                           if jj==0:
                               tgt_image_stack.append(tgt_image.astype('uint8'))
                               src_image_stack.append(src_image.astype('uint8'))
                               
                           src_image_rec_stack[m_name[jj]].append(src_image_rec.astype('uint8'))
                           depth_stack[m_name[jj]].append(depth[n])
                           #plt.figure()
                           #plt.subplot(1,4,1)
                           #plt.imshow(tgt_image.astype('uint8'))
                           #plt.subplot(1,4,2)
                           #plt.imshow(src_image.astype('uint8'))
                           #plt.subplot(1,4,3)
                           #plt.imshow(src_image_rec.astype('uint8'))
                           #plt.subplot(1,4,4)
                           #plt.imshow(rigid_flow[0,:,:,0])
                           print('%s/%s' %(n,N))
                       
            err = err/N
            err_ssim = err_ssim/N
            print(err)
            print(err_ssim)
            
            
save_path = 'S:/Users/Amir/GeoNet/MobileRGBD/video_files_reconst/'          
for n in range(N):

   plt.figure()
   plt.subplots_adjust(top = 0.9, bottom=0.1, hspace=0.1, wspace=0.2)

   plt.subplot(2,6,(2,3))
   plt.imshow(tgt_image_stack[n])
   plt.axis('off')
   plt.title('Input', fontsize=6)
   
   plt.subplot(2,6,(4,5))
   plt.imshow(src_image_stack[n])
   plt.axis('off')
   plt.title('Nearby View', fontsize=6)
   
   plt.subplot(2,6,(7,8))  
   plt.imshow(cv2.medianBlur(src_image_rec_stack['first'][n],5))
   plt.axis('off')
   plt.title('GeoNet-UnSup', fontsize=6)
   
   plt.subplot(2,6,(9,10))
   plt.imshow(cv2.medianBlur(src_image_rec_stack['second'][n],3))
   plt.axis('off')
   plt.title('IndoorGeoNet-UnSup', fontsize=6)
   
   plt.subplot(2,6,(11,12))
   plt.imshow(cv2.medianBlur(src_image_rec_stack['third'][n],3))
   plt.axis('off')
   plt.title('IndoorGeoNet-WSup', fontsize=6)
   
   image_path = save_path + '/%.10d.png' %(n+1)

   plt.savefig(image_path)
   plt.close()