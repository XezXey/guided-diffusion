# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
import argparse

fi = open("/data/supasorn/DECA/CelebAMask-HQ-light-anno.txt", "r")


cluster = 5
outs = [0] * cluster
ws = [0] * cluster
means = []
for i in range(cluster):
  chunk = 2.0 / cluster
  pos = chunk /2 + chunk * i - 1
  means.append(pos)

def gaussian(v, m, c):
  return np.exp(-(v-m)**2 / (2*c*c))
for i, l in enumerate(fi.readlines()):
  print(i)
  sp = l.split(" ")
  light = np.array([float(x) for x in sp[1:]])

  img = cv2.imread("/home/konpat/datasets/CelebAMask-HQ/CelebA-HQ-img/" + sp[0]) / 255.0
  s = np.mean(light[6:9])

  for j in range(cluster):
    w = gaussian(s, means[j], 0.2)
    outs[j] += img * w 
    ws[j] += w
  if i == 3000:
    break

for j in range(cluster):
  outs[j] /= ws[j]

fi.close()
cv2.imwrite("light2.jpg", np.concatenate(outs, 1) * 255)

  # if s < 0:
    # left += img
    # left_count += 1
  # else:
    # right += img
    # right_count += 1
  # stats.append(s)
# print(np.max(stats), np.min(stats), np.mean(stats), np.std(stats))
# for j in np.linspace(-1, 1, 10):
  # print(j, gaussian(j, 1, 0.3))
