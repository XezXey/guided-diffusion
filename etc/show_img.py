import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.isfile(args.file):
        raise(".npz file not found...")
    else:
        data = np.load(args.file)
        lst = data.files
        for item in lst:
            for i in range(len(data[item])):
                # plt.imshow(data[item][i])
                cv2.imwrite("./{}.png".format(i), data[item][i][..., ::-1])
                # print(data[item][i])
                # exit()
                # plt.savefig("./{}.png".format(i))
