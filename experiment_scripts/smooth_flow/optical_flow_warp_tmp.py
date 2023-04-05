import argparse
# import pyflow
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import scipy

parser = argparse.ArgumentParser(
  description="Optical flow debugger by warpping the images")
parser.add_argument('--image_dir', dest='image_dir', help='Input the image directories', required=True)
parser.add_argument('--forward_flow', dest='forward_flow', help='Input the path to forward flow .npy format file', required=True)
parser.add_argument('--backward_flow', dest='backward_flow', help='Input the path to backward flow .npy format file', required=True)
parser.add_argument('--mode', dest='mode', help='Input the mode(input frames by manual or auto) of warping.', required=True)
parser.add_argument('--step_size', dest='step_size', help='Input the step_size(this will work in auto mode)', required=True)
args = parser.parse_args()

def bilinear_interpolation(flow, x_pos, y_pos):
  '''Interpolate (x, y) from values associated with four points.
  Four points are in list of four triplets : (x_pos, y_pos, value)
  - x_pos and y_pos is the pixels position on the images
  - flow is the matrix at x_pos and y_pos to be a reference for interpolated point.
  Ex : x_pos = 12, y_pos = 5.5
       # flow is the matrix of source to be interpolated
       # Need to form in 4 points that a rectangle shape
       flow_location = [(10, 4, 100), === [(x1, y1, value_x1y1),
                        (20, 4, 200), ===  (x2, y1, value_x2y1),
                        (10, 6, 150), ===  (x1, y2, value_x1y2),
                        (20, 6, 300)] ===  (x2, y2, value_x2y2)]
      Reference : https://en.wikipedia.org/wiki/Bilinear_interpolation

  '''
  # Create a flow_location : clip the value from 0-flow.shape[0-or-1]-1
  # x1, y1 : Not lower than 0
  # x2, y2 : Not exceed img size
  x1 = np.clip(int((np.floor(x_pos))), 0, flow.shape[0]-1)
  x2 = np.clip(x1 + 1, 0, flow.shape[0]-1)
  y1 = np.clip(int((np.floor(y_pos))), 0, flow.shape[1]-1)
  y2 = np.clip(y1 + 1, 0, flow.shape[1]-1)
  x_pos = np.clip(x_pos, 0, flow.shape[0]-1)
  y_pos = np.clip(y_pos, 0, flow.shape[1]-1)

  # Last pixels will be the problem that exceed the image size
  if x1 == flow.shape[0]-1:
    x1 = flow.shape[0]-2
  if y1 == flow.shape[0]-1:
    y1 = flow.shape[0]-2

  # print("X : ", x_pos, x1, x2)
  # print("Y : ", y_pos, y1, y2)
  flow_area = [(x1, y1, flow[x1][y1]),
               (x2, y1, flow[x2][y1]),
               (x1, y2, flow[x1][y2]),
               (x2, y2, flow[x2][y2])]

  # print("Flow interesting area : ", flow_area)threshold_slow
  flow_area = sorted(flow_area)
  (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = flow_area

  if x1!=_x1 or x2!= _x2 or y1!=_y1 or y2!=_y2:
    raise ValueError('Given grid do not form a rectangle.')
  if not x1 <= x_pos <= x2 or not y1 <= y_pos <= y2:
    raise ValueError('(x, y) that want to interpolated is not within the rectangle')

  return (q11 * (x2-x_pos) * (y2-y_pos) +
          q21 * (x_pos-x1) * (y2-y_pos) +
          q12 * (x2-x_pos) * (y_pos-y1) +
          q22 * (x_pos-x1) * (y_pos-y1)
          / ((x2-x1) * (y2-y1)) + 0.0)

def get_warping_img(img1, img2, forward_flow, backward_flow):
  print("===>Forward flow for warping : ", forward_flow.shape)
  print("===>Backward flow for warping : ", backward_flow.shape)
  print("\n******************Forward Warping*******************")
  forward_warp_img = forward_warping(img1, img2, backward_flow)
  print("\n******************Backward Warping******************")
  backward_warp_img = backward_warping(img1, img2, forward_flow)
  return forward_warp_img, backward_warp_img


# We apply a inverse warping so the direction will be in the opposite way
# Forward warping a2->a1->ref : Use backward flow from a1<-ref and a2<-a1 then compose to make a "a2<-ref flow". Then make a warp a2->ref using "backward flow"
# Backward warping ref<-a1<-a2 : Use forward flow from ref->a1 and a1->a2 then compose to make a "ref<-a2 flow". Then make a warp a2->ref using "forward flow"
def forward_warping(img1, img2, backward_flow):
  chain_flows = computeChain(backward_flow)
  forward_warp_img = warp_image(img1, chain_flows)
  return forward_warp_img

def backward_warping(img1, img2, forward_flow):
  chain_flows = computeChain(forward_flow)
  backward_warp_img = warp_image(img2, chain_flows)
  return backward_warp_img

# def warp_image(img, chain_flows):
  # warped_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
  # final_warp_flow = chain_flows[-1]
  # warped_img = img
  # print(final_warp_flow)
  # print("Final warp flow : ", final_warp_flow.shape)
  # print("===>Final warp flow : ", chain_flows[-1].shape)
  # print("======>Warp src-image shape : ", img.shape)
  # print("[*]Warping ...", end='')
  # for f in range(chain_flows.shape[0]):
    # for i in range(img.shape[0]): # Rows
      # for j in range(img.shape[1]): # Cols
        # for k in range(img.shape[2]): # RGB channels
          # warped_img[i][j][k] = bilinear_interpolation(warped_img[:, :, k], i + chain_flows[f][i][j][1], j + chain_flows[f][i][j][0])
  # print("Done!")
  # print("======>Warped dest-image shape : ", warped_img.shape)
  # return warped_img

def warp_image(img, chain_flows):
  ''' Warp image function take 2 inputs agruments
      1. img : src image
      2. chain_flows : optical flow(list-like) from cummulative from src to dest images
  '''
  warped_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
  final_warp_flow = chain_flows[-1]
  # print("Max-Min : ", np.max(final_warp_flow), np.min(final_warp_flow))
  print("Final warp flow : ", final_warp_flow.shape)
  print("===>Final warp flow : ", chain_flows[-1].shape)
  print("======>Warp src-image shape : ", img.shape)
  print("[*]Warping ...", end='')
  for i in range(img.shape[0]): # Rows
    for j in range(img.shape[1]): # Cols
      for k in range(img.shape[2]): # RGB channels
        warped_img[i][j][k] = bilinear_interpolation(img[:, :, k], i + final_warp_flow[i][j][1], j + final_warp_flow[i][j][0])
  print("Done!")
  print("======>Warped dest-image shape : ", warped_img.shape)
  return warped_img

def compose_flow(v0, v1):
  ''' This function take 2 flows and compose it
      v0 : ref <- a1
      v1 : a1 <- a2
      Full path : ref <- a1 <- a2
      output : ref <- a2
  '''
  # For compose any 2 adjacent flow together.
  composed_flow = np.zeros((v0.shape[0], v0.shape[1], 2))
  print(composed_flow.shape)
  for i in range(v0.shape[0]):
    for j in range(v0.shape[1]):
      # Find the landed pixels locations on v1 that jump from v0(offset) + i-or-j th pixel
      nr = i + v0[i][j][1]  # On y-axis
      nc = j + v0[i][j][0]  # On x-axis
      # Flow on x-axis
      composed_flow[i][j][0] = v0[i][j][0] + bilinear_interpolation(v1[..., 0], nr, nc)
      # Flow on y-axis
      composed_flow[i][j][1] = v0[i][j][1] + bilinear_interpolation(v1[..., 1], nr, nc)
  print("===>Composed flow : ", composed_flow.shape)
  print("======>V0 : ", v0.shape)
  print("======>V1 : ", v1.shape)
  return composed_flow

def computeChain(flow_list):
  # Compute chain flows from given list of flows
  # flow_list is in flow[i] to flow[j] that want to compose
  # This function won't handle a backward_flow (Need to reverse before pass into flow_list)
  # The output will on the lastest index of flow_chains
  flows_chains = []
  flows_chains.append(flow_list[0]) # Set the start flows
  print("[*]Computing chain flow (Given {} flows)...".format(len(flow_list)))
  for i in range(len(flow_list)-1):
    flows_chains.append(compose_flow(flow_list[i+1], flows_chains[i]))
  return np.array(flows_chains)

# def computeL2Distance(img1, img2, forward_warp_img, backward_warp_img):
  # # Compute L2-distance between src-warped images
  # # L2 - Least squared error
  # forward_warp_distance = np.squared(img2-forward_warp_img)
  # backward_warp_distance = np.squared(img1-backward_warp_img)
  # print("[*]Computing L2-Distance")
  # print("===>Forward warping distance : ", forward_warp_distance)
  # print("===>Backward warping distance : ", backward_warp_distance)

if __name__ == "__main__":
  image_dir = args.image_dir
  forward_flow = np.load(args.forward_flow)
  backward_flow = np.load(args.backward_flow)
  # For re-check this is not the same flow from images
  # print(forward_flow - backward_flow)
  print("=================Warping images using Optical Flow================")
  print("[*]Forward flow ({}) : ".format(forward_flow.shape, forward_flow))
  print("[*]Backward flow ({}) : ".format(backward_flow.shape, backward_flow))
  # Default file pattern : '/raw_frame*.png'
  file_pattern = '/raw_frame*.png'
  image_filelist = sorted(glob.glob(image_dir + file_pattern))
  if args.mode == 'auto':
    # This loop iterate until finishins all of pixles.
    print("[@]Warping forward/backward between 2 given images")
    for i in range(len(image_filelist) - step_size):
      j = i + step_size
      if j >= len(image_filelist):
        exit()
      # Read a img1 and img2
      img1 = cv2.imread(image_filelist[i])[..., ::-1]
      img2 = cv2.imread(image_filelist[j])[..., ::-1]

      # Warping image : Forward warp(img1->img2) and Backward warp(img2->img1)
      forward_warp_img, backward_warp_img = get_warping_img(img1, img2, forward_flow[i:j, ...], backward_flow[i:j, ...])
      # Compute distance between src-warped images
      computeL2Distance(img1, img2, forward_warp_img, backward_warp_img)
      cv2.imwrite('./forward_warp.png', forward_warp_img.astype(int))
      cv2.imwrite('./backward_warp.png', backward_warp_img.astype(int))
      warping_result = np.vstack((np.hstack((img1, img2, forward_warp_img.astype(int))),
                                  np.hstack((img2, img1, backward_warp_img.astype(int)))))#, warped_img))

      plt.imshow(warping_result)
      plt.title("Warping forward/backward between {} and {}".format(i, j))
      plt.show()

  else :
    while True:
      # This loop iterate until finishins all of pixles.
      print("[@]Warping forward/backward between 2 given images")
      i = int(input("[*]Frame i-th of input images : "))
      j = int(input("[*]Frame j-th of input images : "))
      if (i<0) or (j<0):
        exit()

      # Read a img1 and img2
      img1 = cv2.imread(image_filelist[i])[..., ::-1]
      img2 = cv2.imread(image_filelist[j])[..., ::-1]

      # Warping image : Forward warp(img1->img2) and Backward warp(img2->img1)
      forward_warp_img, backward_warp_img = get_warping_img(img1, img2, forward_flow[i:j, ...], backward_flow[i:j, ...])
      # Compute distance between src-warped images
      computeL2Distance(img1, img2, forward_warp_img, backward_warp_img)
      cv2.imwrite('./forward_warp.png', forward_warp_img.astype(int))
      cv2.imwrite('./backward_warp.png', backward_warp_img.astype(int))
      warping_result = np.vstack((np.hstack((img1, img2, forward_warp_img.astype(int))),
                                  np.hstack((img2, img1, backward_warp_img.astype(int)))))#, warped_img))

      plt.imshow(warping_result)
      plt.title("Warping forward/backward between {} and {}".format(i, j))
      plt.show()

