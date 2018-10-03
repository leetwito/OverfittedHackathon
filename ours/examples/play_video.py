# Copyright (C) 2018 Innoviz Technologies
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD 3-Clause license. See the LICENSE file for details.

import os.path as osp
from visualizations.vis import pcshow
import numpy as np
from utilities import data_utils
import os
import csv
import time

if __name__ == '__main__':

    # base_dir = os.path.dirname(os.getcwd())
    # video_dir = os.path.join(base_dir, 'data_examples', 'test_video')

    base_dir = "E:\Datasets\DataHack\World\Train"
    # video_dir = os.path.join(base_dir, 'vid_1')
    video_dir = os.path.join(base_dir, 'vid_2')

    frame_num = data_utils.count_frames(video_dir)
    min_idx = 0
    decimate = 1
    for idx, frame in enumerate(data_utils.enumerate_frames(video_dir)):
        if idx < min_idx or idx % decimate != 0:
            continue
        pc, ego, label = data_utils.read_all_data(video_dir, frame)
        labeled_pc = np.concatenate((pc, label), -1)
        # with open('C:/Users/leetw/PycharmProjects/OverfittedHackathon/ours/example{}.csv'.format(idx), 'w+') as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerows(labeled_pc)
        pcshow(labeled_pc, on_screen_text=osp.join(video_dir, str(frame)), max_points=80000)
        time.sleep(5)

