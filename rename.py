import numpy as np
import os

if __name__ == "__main__":
    directory = "/scratch/slee67/bodycam_npy/"
    [os.rename(os.path.join(directory, f),
               os.path.join(directory, f).replace('-', '_').replace('#', '').replace('–', '_').lower()
               ) for f in os.listdir(directory)]
