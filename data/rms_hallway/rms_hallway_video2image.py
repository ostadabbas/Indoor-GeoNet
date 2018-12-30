# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_raw_loader.py
from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc
import cv2

class rms_hallway_video2image(object):
    def __init__(self, 
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 down_sample=5):
        
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.down_sample = down_sample 
        self.path_ids = ['1', '2','3','4','5','6','7','8','9','10']
        self.hallway_list = ['C1', 'C2', 'C3', 
                          'C4', 'C5','C6']
        
    def collect_video_frames(self):
        
        for hallway_id in self.hallway_list:
            
            drive_dir = os.path.join(self.dataset_dir, hallway_id)
            if os.path.isdir(drive_dir):
                for path_id in self.path_ids:
                    
                    vid_dir = os.path.join(drive_dir, 'videos', path_id)                        
                    vid_path = glob(vid_dir + '/*-v.mp4')[0]
                    cap = cv2.VideoCapture(vid_path)
                    img_dir = os.path.join(drive_dir, 'videos', path_id, 'my_frames')
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    i = 0
                    n_frame = 0
                    while(cap.isOpened()):
                        _, frame = cap.read()
                        if frame is not None:
                            if n_frame % self.down_sample == 0:
                                img_path = img_dir + '\\%.10d.png' %i
                                frame = cv2.resize(frame, (self.img_width, self.img_height))
                                cv2.imwrite(img_path, frame)
                                i += 1
                        else:
                            cap.release()
                        n_frame += 1
                    cap.release()
                    cv2.destroyAllWindows()