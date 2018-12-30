# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 02:25:35 2018

@author: Amir
"""
from glob import glob
import os

dataset_dir = 'root/path/to/rsm/hallway/dataset/' #edit this line

path_ids = ['1'] #['1', '2','3','4','5','6','7','8','9','10']
hallway_list = ['C1'] # ['C1', 'C2', 'C3', 'C4', 'C5','C6']

with open('test_files_mine.txt', 'w') as f:
    
    for hallway_id in hallway_list:
        
        drive_dir = os.path.join(dataset_dir, hallway_id)
        if os.path.isdir(drive_dir):
            for path_id in path_ids:
                img_dir = os.path.join(drive_dir, 'videos', path_id, 'my_frames')
                N = len(glob(img_dir + '/*.png'))
                for n in range(N):
                    frame_id = '%.10d.png' % n
                    img_file = os.path.join(hallway_id,'videos',path_id,'my_frames',frame_id)
                    f.write('%s\n' %img_file)