import numpy as np
import argparse
import glob, os
import cv2
import numpy as np
import tqdm
from PIL import Image

parser = argparse.ArgumentParser(
  description="Optical flow debugger by warpping the images")
parser.add_argument('--image_dir', dest='image_dir', help='Input the image directories', required=True)
parser.add_argument('--flows', dest='flows', help='Input the path to flow .npy format file', required=True)
parser.add_argument('--kpts', dest='kpts', help='Input the path to flow .npy format file', required=True)
parser.add_argument('--video_name', dest='video_name', help='Input the video name', required=True)
parser.add_argument('--save_vis', action='store_true', dest='save_vis', help='Input the path to flow .npy format file', default=False)
parser.add_argument('--save_align', type=str, dest='save_align', default=None)
args = parser.parse_args()

def computeChain(flow_list, init_kpts):
  # Compute chain flows from given list of flows
  # flow_list is in flow[i] to flow[j] that want to compose
  # This function won't handle a backward_flow (Need to reverse before pass into flow_list)
  # The output will on the lastest index of flow_chains
  '''
  flow_list : [flow_1, flow_2, ..., flow_n]
  kpts : [kpt_1, kpt_2, ..., kpt_n]; eact kpt in shape (68, 2)
  '''
  kpts_flow = []  # Store the flow at each keypoints
  for kx, ky in init_kpts:
    # kpts_flow.append(flow_list[0][ky][kx])
    kpts_flow.append(flow_list[0][ky][kx])
  kpts_flow = np.array(kpts_flow)
    
  for i in range(len(flow_list)-1):
    # flows_chains.append(compose_flow_at_kpt(v0=flow_list[i+1], v1=flows_chains[i], kpts=kpts_list[i]))
    update_flow = compose_flow_at_kpt(v0=kpts_flow, v1=flow_list[i+1], kpts=init_kpts)
    kpts_flow = update_flow
    # print(update_flow)
    # exit()
  # print(kpts_flow.shape)
  # print(kpts_flow)
  # exit()
  return kpts_flow

def compose_flow_at_kpt(v0, v1, kpts):
  ''' This function take 2 flows and compose it
      v0 : ref <- a1
      v1 : a1 <- a2
      Full path : ref <- a1 <- a2
      output : ref <- a2
  '''
  # For compose any 2 adjacent flow together.
  composed_flow = np.zeros(shape=kpts.shape)
  # print(composed_flow.shape)
  for idx, kpt in enumerate(kpts):
      # Find the landed pixels locations on v1 that jump from v0(offset) + keypoints position (pixel_ij)
      kx = kpt[0]
      ky = kpt[1]
      # print(i, j, v0.shape)
      nx = kx + v0[idx][0]  # On x-axis
      ny = ky + v0[idx][1]  # On y-axis
      # Flow on x-axis
      # composed_flow[idx][0] = v0[idx][0] + bilinear_interpolation(v1[..., 0], nx, ny)
      composed_flow[idx][0] = v0[idx][0] + bilinear_interp(v1[..., 0], nx, ny)
      # Flow on y-axis
      composed_flow[idx][1] = v0[idx][1] + bilinear_interp(v1[..., 1], nx, ny)
  # print("===>Composed flow : ", composed_flow.shape)
  # print("======>V0 : ", v0.shape)
  # print("======>V1 : ", v1.shape)
  return composed_flow

def warp_kpts(kpts, flows):
  ''' Warp image function take 2 inputs agruments
      1. img : src kpts
      2. chain_flows : optical flow(list-like) from cummulative from src to dest images
  '''
  warped_kpts = np.zeros_like(kpts)
  final_warp_flow = flows
  # print("======> Final warp flow : ", final_warp_flow.shape)
  # print("======> Kpts shape : ", kpts.shape)
  # print("[*] Warping ...", end='')
  for i in range(warped_kpts.shape[0]): # Each kpts
        warped_kpts[i][0] = kpts[i, 0] + final_warp_flow[i][0]
        warped_kpts[i][1] = kpts[i, 1] + final_warp_flow[i][1]
  # print("Done!")
  # print("======>Warped dest-kpts shape : ", warped_kpts.shape)
  return warped_kpts

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
    x1 = np.clip(int((np.floor(x_pos))), 0, flow.shape[1]-1)
    x2 = np.clip(x1 + 1, 0, flow.shape[1]-1)
    y1 = np.clip(int((np.floor(y_pos))), 0, flow.shape[0]-1)
    y2 = np.clip(y1 + 1, 0, flow.shape[0]-1)
    x_pos = np.clip(x_pos, 0, flow.shape[1]-1)
    y_pos = np.clip(y_pos, 0, flow.shape[0]-1)

    # Last pixels will be the problem that exceed the image size
    # if x1 == flow.shape[0]-1:
    #     x1 = flow.shape[0]-2
    # if y1 == flow.shape[0]-1:
    #     y1 = flow.shape[0]-2

    # print("X : ", x_pos, x1, x2)
    # print("Y : ", y_pos, y1, y2)
    flow_area = [(x1, y1, flow[y1][x1]),
                 (x2, y1, flow[y1][x2]),
                 (x1, y2, flow[y2][x1]),
                 (x2, y2, flow[y2][x2])]


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
    
def bilinear_interp(flow, x, y):
    # Get image shape and indices of neighboring pixels
    H, W = flow.shape
    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
    x2, y2 = x1+1, y1+1
    
    # Compute the weights for each pixel based on distance from (x,y)
    w1 = (x2-x)*(y2-y)
    w2 = (x-x1)*(y2-y)
    w3 = (x2-x)*(y-y1)
    w4 = (x-x1)*(y-y1)
    
    # Clip pixel indices to image boundaries
    x1, y1 = np.clip([x1, y1], [0, 0], [W-1, H-1])
    x2, y2 = np.clip([x2, y2], [0, 0], [W-1, H-1])
    
    # Get the neighboring pixels and compute the weighted average
    pixel1 = flow[y1, x1]
    pixel2 = flow[y1, x2]
    pixel3 = flow[y2, x1]
    pixel4 = flow[y2, x2]
    interp_pixel = w1*pixel1 + w2*pixel2 + w3*pixel3 + w4*pixel4
    
    return interp_pixel

if __name__ == "__main__":
  # Load flows
  if os.path.isdir(args.flows):
    flows = {}
    for f in tqdm.tqdm(glob.glob(f'{args.flows}/*.npy')):
      tmp = np.load(f, allow_pickle=True).item()
      # print(tmp.keys())
      # exit()
      flows.update(tmp)
  else:
    flows = np.load(args.flows, allow_pickle=True).item()
  # print(flows.keys())
  # print(len(flows.keys()))
  # exit()
  # flows = np.load(args.flows, allow_pickle=True).tolist()
  # flows = {k: v for d in flows for k, v in d.items()}
    
    
  # Load keypoints
  load_kpts = np.load(args.kpts, allow_pickle=True).item()
  kpts = {}
  for k in load_kpts.keys():
    kpts[k] = np.array(load_kpts[k]['face_landmark'])
  
  n_votes = 3
  
  smooth_kpts = {}
  f_list = sorted(kpts.keys(), key=lambda x:int(x.split('/')[-1][5:-4]))
  for i, f in enumerate(f_list):  # Process each frames
    
    # Get frames index that used as voters
    fw_voters = np.clip(i-n_votes, 0, len(f_list))
    fw_v_list = f_list[fw_voters:i]
    
    bw_voters = np.clip(i+n_votes, 0, len(f_list))
    bw_v_list = f_list[i+1:bw_voters+1]
    
    # Get the flows for specifics frame
    if len(fw_v_list) != 0:
      fw_kpts_list = fw_v_list  # Exclude the current frame
      fw_v_list = fw_v_list + [f]
    if len(bw_v_list) != 0:
      bw_kpts_list = bw_v_list  # Exclude the current frame
      bw_v_list = [f] + bw_v_list
      
    print("#" * 77)
    print("[#] Frame : ", f)
    print("[#] fw_flows_voters (use forward flow): ", fw_v_list)
    print("[#] bw_flows_voters (use backward flow) : ", bw_v_list)
    
    fw_v_list = [f"{fw_v_list[i].split('.')[0]}_{fw_v_list[i+1].split('.')[0]}" for i in range(len(fw_v_list)-1)]
    bw_v_list = [f"{bw_v_list[i].split('.')[0]}_{bw_v_list[i+1].split('.')[0]}" for i in range(len(bw_v_list)-1)]
    
    # Load only usaged flows
    
    print("[#] fw_flows_voters (use forward flow) : ", fw_v_list)
    print("[#] bw_flows_voters (use backward flow): ", bw_v_list)
    
    candi_kpts = [kpts[f]]
    
    #NOTE: Use for moving the keypoints from t => t+1
    flows_fw = []
    for k in fw_v_list:
      if k in flows.keys():
        flows_fw.append(flows[k]['fw'])
      
    if len(flows_fw) != 0:
      #NOTE: Forward flows = warping from [t-n_votes, ..., t-2, t-1] ===> to "t" kpts
      fw_out = []
      for i, k in enumerate(fw_kpts_list):
        chain = flows_fw[:i+1]
        if len(chain) > 1:
          chain = computeChain(chain, kpts[k])
        else:
          tmp = []
          for kx, ky in kpts[k]:
            tmp.append(chain[0][ky][kx])
          chain = np.array(tmp)
        out = warp_kpts(kpts[k], chain)
        candi_kpts.append(out)
    
    #NOTE: Use for moving the keypoints from t+1 => t
    flows_bw = []
    for k in bw_v_list:
      if k in flows.keys():
        flows_bw.append(flows[k]['bw'])
      
    if len(flows_bw) != 0:
      #NOTE: Backward flows = warping from [t+1, t+2, ..., t+n_votes] ===> to "t" kpts
      fw_out = []
      for i, k in enumerate(bw_kpts_list[::-1]):
        chain = flows_bw[:i+1]
        if len(chain) > 1:
          chain = computeChain(chain[::-1], kpts[k])
        else:
          tmp = []
          for kx, ky in kpts[k]:
            tmp.append(chain[0][ky][kx])
          chain = np.array(tmp)
        out = warp_kpts(kpts[k], chain)
        candi_kpts.append(out)
    
    candi_kpts = np.stack(candi_kpts, axis=0)
    print(f"{f} : shape={candi_kpts.shape}, mean={np.mean(candi_kpts, axis=0).shape}, sd={np.std(candi_kpts, axis=0).shape}")
    smooth_kpts[f] = np.mean(candi_kpts, axis=0)
    
  # print(smooth_kpts.keys())
  
  if args.save_align is not None:
    print("=" * 77)
    print("=" * 77)
    print("[#] Aligning w/ new kpts")
    import align_lib
    import shutil
    data_dir = f'/data/mint/DPM_Dataset/Videos/{args.video_name}/images/'
    frames = sorted(kpts.keys(), key=lambda x:int(x[5:-4]))
    save_path = f'{args.save_align}/aligned_images/valid/'
    copy_path = f'{args.save_align}/images/'
    os.makedirs(args.save_align, exist_ok=True)
    os.makedirs(copy_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    align_params = {}
    for f in tqdm.tqdm(frames):
      aligned_img, aligned_dict = align_lib.image_align(src_file=data_dir + f,
                                                        face_landmarks=smooth_kpts[f], 
                                                        output_size=256)
      align_params[f] = aligned_dict
      align_params[f]['face_landmark_smooth'] = smooth_kpts[f]
      aligned_img.save(f'{save_path}/{f}')
      shutil.copyfile(src=data_dir + f, dst=copy_path + f)
    np.save(file=f'{args.save_align}/{args.video_name}_align_params.npy', arr=align_params)
    print("[#] Done!!!")
    print(f"Save params at {args.save_align}/{args.video_name}_align_params.npy")
    print(f"Save images at {save_path}")
  
  if args.save_vis:
    # Save the visualization
    
    import align_lib
    path = f'./examples/{args.video_name}/'
    os.makedirs(path, exist_ok=True)
    data_dir = f'/data/mint/DPM_Dataset/Videos/{args.video_name}/images/'
    frames = sorted(kpts.keys(), key=lambda x:int(x[5:-4]))
    
    aligned_vis = []
    for f in frames:
        aligned_img = align_lib.image_align(src_file=data_dir + f,
                                  face_landmarks=load_kpts[f]['face_landmark'], 
                                  output_size=256)
        aligned_vis.append(aligned_img)

    align_lib.save_video(aligned_vis, path=f'{path}/aligned.mp4', fps=25)
    
    
    aligned_vis = []
    for f in frames:
        aligned_img = align_lib.image_align(src_file=data_dir + f,
                                  face_landmarks=smooth_kpts[f], 
                                  output_size=256)
        aligned_vis.append(aligned_img)

    align_lib.save_video(aligned_vis, path=f'{path}/smth_aligned_fixed.mp4', fps=25)
  
    kpts_vis = []
    for f in frames:
        img = np.array(Image.open(data_dir + f))
        lmk = align_lib.draw_face_landmarks(img, kpts[f])
        kpts_vis.append(lmk)

    align_lib.save_video(kpts_vis, path=f'{path}/lmk.mp4')
  
    kpts_vis = []
    for f in frames:
        img = np.array(Image.open(data_dir + f))
        lmk = align_lib.draw_face_landmarks(img, smooth_kpts[f])
        kpts_vis.append(lmk)

    align_lib.save_video(kpts_vis, path=f'{path}/smth_lmk.mp4')
  
    kpts_vis = []
    for f in frames:
        img = np.array(Image.open(data_dir + f))
        lmk = align_lib.draw_face_landmarks(img, kpts[f], smooth_kpts[f])
        kpts_vis.append(lmk)

    align_lib.save_video(kpts_vis, path=f'{path}/cmp_lmk.mp4')
  