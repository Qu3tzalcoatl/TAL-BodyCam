import shutil
import numpy as np

file_list = np.load('file_list_v2.npy')
src = "./videos/Chicago_raw/"
dest = "/scratch/slee67/bodycam_npy_v2/"
new_file_list = []

for file in file_list:
    new_file = file.replace('#', '').replace('-', '_').replace('–', '_')
    if not os.path.exists(dest+new_file):
        shutil.copy2(src+file, dest+new_file)
    new_file_list.append(new_file)

np.save("./new_file_list.npy", new_file_list)






