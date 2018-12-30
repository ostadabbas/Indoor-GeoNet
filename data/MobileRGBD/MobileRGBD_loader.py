# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_raw_loader.py
from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc
import cv2

class MobileRGBD_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split,
                 img_height=128,
                 img_width=416,
                 seq_length=5,
                 remove_static=True):
        dir_path = os.path.dirname(os.path.realpath(__file__))        
        test_scene_file = dir_path + '/test_scenes_' + split + '.txt'
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        #self.path_ids = ['1', '2','3','4','5','6','7','8','9','10']
        #self.hallway_list = ['C1', 'C2', 'C3', 
        #                 'C4', 'C5','C6']
        
        excluded_subfolders = ['1stBodyCorridor',
                       '2ndBodyCorridor',
                       'Traj_54_0_Corridor_0.3',
                       'Traj_54_15_Corridor_0.3',
                       'Traj_54_-15_Corridor_0.3',
                       'Traj_54_30_Corridor_0.3',
                       'Traj_132_0_Corridor_0.3',
                       'Traj_132_15_Corridor_0.3',
                       'Traj_132_-15_Corridor_0.3',
                       'Traj_132_-30_Corridor_0.3']

        subfolders = []
        for subfolder in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, subfolder)):
                subfolders.append(subfolder)
        
        for ex_subfolder in excluded_subfolders:
            subfolders.remove(ex_subfolder)
            
        self.hallway_list = subfolders
        
        
        if remove_static:
            static_frames_file = dir_path + '/static_frames.txt'
            self.collect_static_frames(static_frames_file)
        self.collect_train_frames(remove_static)

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = []
        for fr in frames:
            if fr == '\n':
                continue
            hallway_id, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            self.static_frames.append(hallway_id, ' ', curr_fid)
        
    def collect_train_frames(self, remove_static):
        all_frames = []
        for hallway_id in self.hallway_list:
            
            drive_dir = os.path.join(self.dataset_dir, hallway_id)
            if os.path.isdir(drive_dir):
                img_dir = os.path.join(drive_dir, 'video')
                N = len(glob(img_dir + '/*.png'))
                for n in range(N):
                    frame_id = '%.10d' % n
                    all_frames.append(hallway_id + ' ' + frame_id)
                        
        if remove_static:
            for s in self.static_frames:
                try:
                    all_frames.remove(s)
                except:
                    pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_hallway, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_hallway, _ = frames[min_src_idx].split(' ')
        max_src_hallway, _ = frames[max_src_idx].split(' ')
        if tgt_hallway == min_src_hallway and tgt_hallway == max_src_hallway:
            return True
        return False

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_hallway, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(curr_hallway, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y
    
    def load_depth_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        depth_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_hallway, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_depth_raw(curr_hallway, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = cv2.resize(curr_img, (self.img_width, self.img_height))
            depth_seq.append(curr_img)
        return depth_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        depth_seq, _, _ = self.load_depth_sequence(frames, tgt_idx, self.seq_length)
        tgt_hallway, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics_raw(tgt_hallway, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['depth_seq'] = depth_seq
        example['folder_name'] = tgt_hallway + '/'
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, hallway_id, frame_id):
        
        img_file = os.path.join(self.dataset_dir, hallway_id, 'video', frame_id + '.png')
        img = scipy.misc.imread(img_file)
        return img

    def load_depth_raw(self, hallway_id, frame_id):
        
        depth_file = os.path.join(self.dataset_dir, hallway_id, 'depth', frame_id + '.png')
        #depth_temp = scipy.misc.imread(depth_file)
        depth_temp = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        
        depth = np.zeros((1080,1920)).astype('uint16')
        depth_temp = cv2.resize(depth_temp,(1305,1080))
        #depth_temp = np.uint16(scipy.misc.imresize(depth_temp, (1080, 1305)))
        depth[:,0+270:1305+270] = depth_temp
        
        return depth

    def load_intrinsics_raw(self, hallway_id, frame_id):
        
        calib_file = os.path.join(self.dataset_dir, 'calib_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect'], (3, 4))
        intrinsics = P_rect[:3, :3]
        return intrinsics

    def read_raw_calib_file(self,filepath):
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

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out


