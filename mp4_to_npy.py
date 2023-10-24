import skvideo.io
import numpy as np
from npy_append_array import NpyAppendArray
import os
from pathlib import Path


file_list = np.load('file_list_v2.npy')
video_dir = "./videos/Chicago_raw/"
npy_dest = "/scratch/slee67/bodycam_npy_v2/"
new_file_list = []

for file_ext in file_list:
    print("Working on file: "+file_ext)
    start_path = video_dir + file_ext
    file_name = Path(start_path).stem

    new_file_name = file_name.replace('-', '_').replace('#', '').replace('–', '_')
    end_path = npy_dest + new_file_name + ".npy"
    new_file_list.append(new_file_name)

    if not os.path.exists(end_path):
        videodata = skvideo.io.vreader(start_path)
        with NpyAppendArray(end_path) as npaa:
            flag = False
            for frame in videodata:
                if flag:
                    npaa.append(test)
                else:
                    test = np.stack((frame, frame), axis=0)
                    print(test.shape)
                    npaa.append(test)
                    flag = True
                # break
            print("SHAPE: ")
            print(npaa.shape)
        print("File saved!\n")
np.save(new_file_list, "file_list_UTF8.npy")
