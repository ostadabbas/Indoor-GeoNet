# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 02:25:35 2018

@author: Amir
"""
from glob import glob
import os

dataset_dir = 'root/path/to/MobileRGBD/dataset/' #edit this line

#path_ids = ['1'] #['1', '2','3','4','5','6','7','8','9','10']
#hallway_list = ['C1'] # ['C1', 'C2', 'C3', 'C4', 'C5','C6']

#hallway_list = ['Traj_132_-15_Corridor_0.7',
#                'Traj_132_-15_Corridor_1.1',
#                'Traj_132_-30_Corridor_0.7',
#                'Traj_132_-30_Corridor_1.1',
#                'Traj_132_0_Corridor_0.7',
#                'Traj_132_0_Corridor_1.1',
#                'Traj_132_15_Corridor_0.7',
#                'Traj_132_15_Corridor_1.1',
#                'Traj_54_-15_Corridor_0.7',
#                'Traj_54_-15_Corridor_1.1',
#                'Traj_54_0_Corridor_0.7',
#                'Traj_54_0_Corridor_1.1',
#                'Traj_54_15_Corridor_0.7',
#                'Traj_54_15_Corridor_1.1',
#                'Traj_54_30_Corridor_0.7',
#                'Traj_54_30_Corridor_1.1']
hallway_list = ['Traj_54_0_Corridor_0.7']

with open('test_files_mine.txt', 'w') as f:
    
    for hallway_id in hallway_list:
        
        drive_dir = os.path.join(dataset_dir, hallway_id, 'video')
        if os.path.isdir(drive_dir):

            N = len(glob(drive_dir + '/*.png'))
            for n in range(N):
                frame_id = '%.10d.png' % n
                img_file = os.path.join(hallway_id,'video',frame_id)
                f.write('%s\n' %img_file)