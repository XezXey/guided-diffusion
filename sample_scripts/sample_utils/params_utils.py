import numpy as np
import pandas as pd
import scipy.ndimage
import torch as th
import glob, os, sys
import cv2
from collections import defaultdict

def params_to_model(shape, exp, pose, cam, lights):

    from model_3d.FLAME import FLAME
    from model_3d.FLAME.config import cfg as flame_cfg
    from model_3d.FLAME.utils.renderer import SRenderY
    import model_3d.FLAME.utils.util as util

    flame = FLAME.FLAME(flame_cfg.model).cuda()
    verts, landmarks2d, landmarks3d = flame(shape_params=shape, 
            expression_params=exp, 
            pose_params=pose)
    renderer = SRenderY(image_size=256, obj_filename=flame_cfg.model.topology_path, uv_size=flame_cfg.model.uv_size).cuda()

    ## projection
    landmarks2d = util.batch_orth_proj(landmarks2d, cam)[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
    landmarks3d = util.batch_orth_proj(landmarks3d, cam); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
    trans_verts = util.batch_orth_proj(verts, cam); trans_verts[:,:,1:] = -trans_verts[:,:,1:]

    ## rendering
    shape_images = renderer.render_shape(verts, trans_verts, lights=lights)

    # opdict = {'verts' : verts,}
    # os.makedirs('./rendered_obj', exist_ok=True)
    # save_obj(renderer=renderer, filename=(f'./rendered_obj/{i}.obj'), opdict=opdict)
    
    return {"shape_images":shape_images, "landmarks2d":landmarks2d, "landmarks3d":landmarks3d}

def save_obj(renderer, filename, opdict):
    '''
    vertices: [nv, 3], tensor
    texture: [3, h, w], tensor
    '''
    import model_3d.FLAME.utils.util as util
    i = 0
    vertices = opdict['verts'][i].cpu().numpy()
    faces = renderer.faces[0].cpu().numpy()
    colors = np.ones(shape=vertices.shape) * 127.5

    # save coarse mesh
    util.write_obj(filename, vertices, faces, colors=colors)

# def get_R_normals(n_step):
#     src = np.array([0, 0, 2.50])
#     dst = np.array([0, 0, 6.50])
#     rvec = np.linspace(src, dst, n_step)
#     R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
#     R = np.stack(R, axis=0)
#     return R

def get_R_normals(n_step):
    if n_step % 2 == 0:
        fh = sh = n_step//2
    else:
        fh = int(n_step//2)
        sh = fh + 1
        
    src = np.array([0, 6.50, 0])
    # dst = np.array([0, 2.50, 0])
    dst = np.array([0, 1.00, 0])
    rvec_f = np.linspace(src, dst, fh)
    
    src = rvec_f[-1]
    # dst = np.array([0, rvec_f[-1][1], -8.00])
    dst = np.array([0, 2.50, -8.00])
    rvec_s = np.linspace(rvec_f[-1], dst, sh)
    # print(rvec_f.shape, rvec_s.shape)
    rvec = np.concatenate((rvec_f, rvec_s), axis=0)
    # print(rvec_f)
    # print(rvec_s)
    # print(rvec)
    # print(rvec.shape)
    R = [cv2.Rodrigues(rvec[i])[0] for i in range(rvec.shape[0])]
    R = np.stack(R, axis=0)
    return R

def grid_sh(n_grid, sh=None, sx=[-1, 1], sy=[1, 0], sh_scale=1.0, use_sh=False):
    sh_light = []
    sh_original = sh.cpu().numpy().copy().reshape(-1, 9, 3)
    print(f"[#] Buiding grid sh with : span_x={sx}, span_y={sy}, n_grid={n_grid}")
    print(f"[#] Given sh : \n{sh_original}")
    # sx is from left(negative) -> right(positive)
    # sy is from top(positive) -> bottom(negative)
    # print(sx, sy)
    # print(np.linspace(sx[0], sx[1], n_grid))
    # exit()
    for ix, lx in enumerate(np.linspace(sx[0], sx[1], n_grid)):
        for iy, ly in enumerate(np.linspace(sy[0], sy[1], n_grid)):
            l = np.array((lx, ly, 1))
            l = l / np.linalg.norm(l)
            
            if use_sh:
                tmp_light = sh_original.copy()
            else:
                tmp_light = np.zeros((1, 9, 3))
                tmp_light[0:1, 0:1, :] = sh_original[0:1, 0:1, :] * sh_scale
                
            # if iy in [1, 2, 3]:
                # print("IN", iy)
                # print(tmp_light)
                # tmp_light = tmp_light * sh_scale 
                # print(tmp_light)
            
            tmp_light[0:1, 1:2, :] = l[0]
            tmp_light[0:1, 2:3, :] = l[1]
            tmp_light[0:1, 3:4, :] = l[2]
            # if iy in [0, 1, 2, 3]:
            tmp_light = tmp_light * sh_scale 
            sh_light.append(tmp_light)
        # exit()
    sh_light = np.concatenate(sh_light, axis=0)
    sh_light = np.concatenate((sh_original.reshape(-1, 9, 3), sh_light))
    print(f"[#] Out grid sh : \n{sh_light.shape}")
    return sh_light

def load_flame_mask(parts=['face']):
    if os.path.exists('/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl') and os.path.exists('/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl'):
        f_mask = np.load('/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
        v_mask = np.load('/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
    elif os.path.exists('/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl') and os.path.exists('/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl'):
        f_mask = np.load('/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
        v_mask = np.load('/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
    elif os.path.exists('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl') and os.path.exists('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl'):
        f_mask = np.load('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks_face-id.pkl', allow_pickle=True, encoding='latin1')
        v_mask = np.load('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/data/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
    else:
        print("[#] No FLAME_masks found!")
        exit()
    mask={
        'v_mask':sum([v_mask[part].tolist() for part in parts], []),
        'f_mask':sum([f_mask[part].tolist() for part in parts], [])
    }
    return mask        

def init_deca(useTex=False, extractTex=True, device='cuda', 
              deca_mode='only_renderer', mask=None, deca_obj=None, rasterize_type='standard'):
    
    # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
    # sys.path.insert(1, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')
    sys.path.append('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
    sys.path.insert(1, '/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')

    from decalib import deca
    from decalib.utils.config import cfg as deca_cfg
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterize_type
    deca_cfg.model.extract_tex = extractTex
    deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    return deca_obj

def sh_to_ld_brightest_region(sh):
    from scipy.spatial.transform import Rotation as R
    from matplotlib import pyplot as plt
    import cv2

    import pyshtools as pysh

    def applySHlight(normal_images, sh_coeff):
        N = normal_images
        sh = th.stack(
            [
            N[0] * 0.0 + 1.0,
            N[0],
            N[1],
            N[2],
            N[0] * N[1],
            N[0] * N[2],
            N[1] * N[2],
            N[0] ** 2 - N[1] ** 2,
            3 * (N[2] ** 2) - 1,
            ],
            0,
        )  # [9, h, w]
        pi = np.pi
        constant_factor = th.tensor(
            [
            1 / np.sqrt(4 * pi),
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ]
        ).float()
        sh = sh * constant_factor[:, None, None]

        shading = th.sum(
            sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
        )  # [9, 3, h, w]
        return shading

    def applySHlightXYZ(xyz, sh):
        out = applySHlight(xyz, sh)
        # out /= pt.max(out)
        out *= 0.7
        return th.clip(out, 0, 1)

    def genSurfaceNormals(n):
        x = th.linspace(-1, 1, n)
        y = th.linspace(1, -1, n)
        y, x = th.meshgrid(y, x)

        z = (1 - x ** 2 - y ** 2)
        z[z < 0] = 0
        z = th.sqrt(z)
        return th.stack([x, y, z], 0)

    def drawSphere(sh, img_size=256):
        n = img_size
        xyz = genSurfaceNormals(n)
        out = applySHlightXYZ(xyz, sh)
        out[:, xyz[2] == 0] = 0
        return out, xyz
    
    sh = sh.reshape(-1, 9, 3)[0]
    visible_sphere, normals_sphere = drawSphere(sh)
    visible_sphere = visible_sphere.permute(1, 2, 0).cpu().numpy()
    plt.imshow(visible_sphere)
    plt.savefig('./visible_sphere.png')
    # Locate the brightest point
    visible_sphere_draw = (visible_sphere.copy() * 255).astype(np.uint8)
    
    # Gray scale
    visible_sphere_tmp = cv2.cvtColor(visible_sphere_draw, cv2.COLOR_RGB2GRAY)
    brightest = np.unravel_index(visible_sphere_tmp.argmax(), visible_sphere_tmp.shape)
    
    bright_area = np.isclose(a=visible_sphere_tmp, b=visible_sphere_tmp.max(), atol=50)
    import scipy
    cm = scipy.ndimage.center_of_mass(bright_area)
    
    # print(bright_area.shape)
    bright_area = np.repeat(bright_area[..., None] * 255, 3, axis=2).astype(np.uint8)
    cv2.circle(bright_area, (int(cm[1]), int(cm[0])), 5, (255, 0, 0), -1)
    plt.imshow(bright_area)
    plt.savefig('./bright_area.png')
    
    cv2.circle(visible_sphere_draw, (int(cm[1]), int(cm[0])), 5, (0, 255, 0), -1)
    plt.imshow(visible_sphere_draw)
    plt.savefig('./visible_sphere_draw.png')
    
    normals_sphere = normals_sphere.permute(1, 2, 0).cpu().numpy()
    light_direction = normals_sphere[brightest[0], brightest[1]]
    light_direction = normals_sphere[int(cm[0]), int(cm[1])]
    
    return th.tensor(light_direction)[None, ...], (normals_sphere, visible_sphere, visible_sphere_draw, bright_area)

def sh_to_ld_brightest_region_G(sh, scale=0.03):
    from scipy.spatial.transform import Rotation as R
    from matplotlib import pyplot as plt
    import cv2

    import pyshtools as pysh

    def applySHlight(normal_images, sh_coeff):
        N = normal_images
        sh = th.stack(
            [
            N[0] * 0.0 + 1.0,
            N[0],
            N[1],
            N[2],
            N[0] * N[1],
            N[0] * N[2],
            N[1] * N[2],
            N[0] ** 2 - N[1] ** 2,
            3 * (N[2] ** 2) - 1,
            ],
            0,
        )  # [9, h, w]
        pi = np.pi
        constant_factor = th.tensor(
            [
            1 / np.sqrt(4 * pi),
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
            ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
            (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ]
        ).float()
        sh = sh * constant_factor[:, None, None]

        shading = th.sum(
            sh_coeff[:, :, None, None] * sh[:, None, :, :], 0
        )  # [9, 3, h, w]
        return shading

    def applySHlightXYZ(xyz, sh):
        out = applySHlight(xyz, sh)
        # out /= pt.max(out)
        out *= 0.7
        return th.clip(out, 0, 1)

    def genSurfaceNormals(n):
        x = th.linspace(-1, 1, n)
        y = th.linspace(1, -1, n)
        y, x = th.meshgrid(y, x)

        z = (1 - x ** 2 - y ** 2)
        z[z < 0] = 0
        z = th.sqrt(z)
        return th.stack([x, y, z], 0)

    def drawSphere(sh, img_size=256):
        n = img_size
        xyz = genSurfaceNormals(n)
        out = applySHlightXYZ(xyz, sh)
        out[:, xyz[2] == 0] = 0
        return out, xyz
    
    sh = sh.reshape(-1, 9, 3)[0]
    visible_sphere, normals_sphere = drawSphere(sh)
    mask = (normals_sphere[2:3].permute(1, 2, 0) != 0).cpu().numpy()
    visible_sphere = visible_sphere.permute(1, 2, 0).cpu().numpy()

    # plt.imshow(visible_sphere)
    # plt.savefig('./visible_sphere.png')

    # Gray scale
    visible_sphere_proc = (visible_sphere.copy() * 255).astype(np.uint8)
    visible_sphere_proc = cv2.cvtColor(visible_sphere_proc, cv2.COLOR_RGB2GRAY)
    sphere_intensity = visible_sphere_proc.copy() / 255.0
    import scipy

    def gaussian(x, mean, sigma):
        # Define the Gaussian function
        return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    # Use max intensity as mean for representing the brightest point
    max_as_mean = np.max(sphere_intensity)
    sigma = max_as_mean * scale  # You can adjust this value for spread

    sphere_to_gaussian = gaussian(sphere_intensity, max_as_mean, sigma)[..., None]

    total_mass = np.sum(sphere_to_gaussian * mask)
    assert total_mass > 0
    h, w, _ = sphere_to_gaussian.shape
    grid = np.meshgrid(np.arange(w), np.arange(h))

    x_centroid = np.sum(grid[0][..., None] * sphere_to_gaussian * mask) / total_mass  # vertical axis
    y_centroid = np.sum(grid[1][..., None] * sphere_to_gaussian * mask) / total_mass  # horizontal axis
    cm = scipy.ndimage.center_of_mass(sphere_to_gaussian * mask)

    # Visualize the original image and the Gaussian distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sphere_intensity, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sphere_to_gaussian, cmap='gray')
    plt.axhline(cm[0], color='red', linestyle='--')
    plt.axvline(cm[1], color='red', linestyle='--')
    plt.title('Scipy (mu={:.2f}, f={:.3f}, std={:.2f})'.format(max_as_mean, scale, sigma))

    plt.subplot(1, 3, 3)
    plt.imshow(sphere_to_gaussian, cmap='gray')
    plt.axhline(y_centroid, color='blue', linestyle='--')
    plt.axvline(x_centroid, color='blue', linestyle='--')
    plt.title('W/ mask (mu={:.2f}, f={:.3f}, std={:.2f})'.format(max_as_mean, scale, sigma))
    plt.axis('off')
    fig = plt.gcf()
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    canvas = FigureCanvas(fig)
    canvas.draw()
    # Convert canvas to an image array
    image_sph_ld = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8) # H x W x 3
    image_sph_ld = image_sph_ld.reshape(canvas.get_width_height()[::-1] + (3,)) # H x W x 3
    plt.close()

    normals_sphere = normals_sphere.permute(1, 2, 0).cpu().numpy()
    # light_direction = normals_sphere[int(cm[0]), int(cm[1])]
    light_direction = normals_sphere[int(y_centroid), int(x_centroid)]
    
    return th.tensor(light_direction)[None, ...], (normals_sphere, visible_sphere, visible_sphere_proc, image_sph_ld)

def sh_to_ld(sh):
    #NOTE: Roughly Convert the SH to light direction
    sh = sh.reshape(-1, 9, 3)
    ld = th.mean(sh[0:1, 1:4, :], dim=2)
    return ld

def render_shadow_mask(sh_light, cam, verts, deca, axis_1=False):
    sys.path.append('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
    sys.path.append('/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
    sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')

    from decalib.utils import util
    
    shadow_mask_all = []
    if verts.shape[0] >= 2:
        tmp = []
        for i in range(1, verts.shape[0]):
            tmp.append(th.allclose(verts[[0]], verts[[i]]))
        assert all(tmp)
        
    depth_image, alpha_image = deca.render.render_depth(verts.cuda())   # Depth : B x 1 x H x W
    _, _, h, w = depth_image.shape
    depth_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
    depth_grid = np.repeat(np.stack((depth_grid), axis=-1)[None, ...], repeats=sh_light.shape[0], axis=0)   # B x H x W x 2
    depth_grid = np.concatenate((depth_grid, depth_image.permute(0, 2, 3, 1)[..., 0:1].cpu().numpy()), axis=-1) # B x H x W x 3
    depth_grid[..., 2] *= 256
    depth_grid = th.tensor(depth_grid).cuda()
    shadow_mask = th.clone(depth_grid[:, :, :, 2])
    # print(shadow_mask.shape, sh_light.shape)
    for i in range(sh_light.shape[0]):
        each_depth_grid = depth_grid[i].clone()
        #NOTE: Render the shadow mask from light direction
        ld = sh_to_ld(sh=th.tensor(sh_light[[i]])).cuda()
        ld = util.batch_orth_proj(ld[None, ...], cam[None, ...].cuda());     # This fn takes pts=Bx3, cam=Bx3
        ld[:, :, 1:] = -ld[:, :, 1:]
        ray = ld.view(3).cuda()
        if axis_1:
            ray[1] *= -1    # This for jst temporarly fix the axis 1 which the shading is bright in the middle, but the light direction is back of the head
        ray[2] *= 0.5
        n = 256
        ray = ray / th.norm(ray)
        mxaxis = max(abs(ray[0]), abs(ray[1]))
        shift = ray / mxaxis * th.arange(n).view(n, 1).cuda()
        coords = each_depth_grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)

        output = th.nn.functional.grid_sample(
            th.tensor(np.tile(each_depth_grid[:, :, 2].view(1, 1, n, n).cpu().numpy(), [n, 1, 1, 1])).cuda(),
            coords[..., :2] / (n - 1) * 2 - 1,
            align_corners=True)
        diff = coords[..., 2] - output[:, 0] 
        shadow_mask[i] *= (th.min(diff, dim=0)[0] > -0.1) * 0.5 + 0.5
        
    return th.clip(shadow_mask, 0, 255.0)/255.0

def render_shadow_mask_with_smooth(sh_light, cam, verts, deca, rt_dict, use_sh_to_ld_region=True, axis_1=False, up_rate=1, device='cuda', org_h=128, org_w=128):
    print("[#] Rendering shadow mask with smooth (pertubation version)...")
    sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')
    sys.path.append('/home/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
    sys.path.append('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
    from decalib.utils import util
    from torchvision.transforms import GaussianBlur
    import cv2, tqdm
    
    if verts.shape[0] >= 2:
        tmp = []
        for i in range(1, verts.shape[0]):
            tmp.append(th.allclose(verts[[0]], verts[[i]]))
        assert all(tmp)

    verts = verts[0:1, ...].cuda()  # Save memory by rendering only 1st image
    # depth_image_f, _ = deca['face'].render.render_depth(verts, up_rate=up_rate)   # Depth : B x 1 x H x W
    # depth_image_fe, _ = deca['face_eyes'].render.render_depth(verts, up_rate=up_rate)   # Depth : B x 1 x H x W
    depth_image_fs, _ = deca['face_scalp'].render.render_depth(verts, up_rate=up_rate)   # Depth : B x 1 x H x W
    B, C, h, w = depth_image_fs.shape

    depth_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
    depth_grid = np.stack((depth_grid), axis=-1)[None, ...]   # 1 x H x W x 2

    orig = depth_image_fs.detach().clone()# * mask_eyes  # Remove depth of eyes out
    mask_face = orig != 0
    # mask_eyes = (depth_image_f!=0) + (depth_image_fe==0)
    # mask_face = ((depth_image_f != 0) + (depth_image_fe == 0)) * (depth_image_fs != 0)
    mask_face = mask_face.float()

    blurred_orig = GaussianBlur(kernel_size=13, sigma=2.0)(orig)#.cpu().numpy()
    assert blurred_orig.shape == depth_image_fs.shape

    blurred_mask = GaussianBlur(kernel_size=13, sigma=2.0)(mask_face)#.cpu().numpy()
    blurred_mask[blurred_mask == 0] = 1e-10
    blurred_orig = blurred_orig / blurred_mask

    blurred_orig = blurred_orig * (mask_face)

    # depth_image_smooth = blurred_orig * 100
    print(f"[#] Scale depth with {rt_dict['scale_depth']}...")
    depth_image_smooth = blurred_orig * rt_dict['scale_depth']

    depth_grid = np.concatenate((depth_grid, depth_image_smooth.permute(0, 2, 3, 1)[..., 0:1].cpu().numpy()), axis=-1) # B x H x W x 3

    # depth_grid[..., 2] *= (256 * up_rate)
    depth_grid = th.tensor(depth_grid).to(device)   # B x H x W x 3; B should be 1
    assert depth_grid.shape[0] == 1
    depth_grid = depth_grid.squeeze(0)

    out_shadow_mask = []
    out_kk = []
    out_sph_ld = []
    rt_scale = rt_dict['rt_regionG_scale']
    if use_sh_to_ld_region:
        print(f"[#] Approximate the light direction from brightest spot with rt_scale={rt_scale}...")
    else:
        print(f"[#] Approximate the light direction from the mean of SH coffecients [2nd, 3rd and 4th] ...")
    for i in tqdm.tqdm(range(sh_light.shape[0]), position=0, desc="Rendering Shadow Mask on each Sh..."):
        #NOTE: Render the shadow mask from light direction

        shadow_mask = th.clone((depth_image_fs.permute(0, 2, 3, 1)[:, :, :, 0])).to(device)
        if use_sh_to_ld_region:
            ld, misc_dat = sh_to_ld_brightest_region_G(sh=th.tensor(sh_light[[i]]), scale=rt_scale)  # Output shape = (1, 3)
            image_sph_ld = misc_dat[-1]
            ld[:, 2] *= -1
        else:
            ld = sh_to_ld(sh=th.tensor(sh_light[[i]]))  # Output shape = (1, 3)
            image_sph_ld = None
        ld = util.batch_orth_proj(ld[None, ...].cuda(), cam[None, ...].cuda());     # This fn takes pts=Bx3, cam=Bx3
        ld[:, :, 1:] = -ld[:, :, 1:]
        ray = ld.view(3).to(device)
        ray = ray / th.norm(ray)

        if axis_1:
            ray[1] *= -1    # This for jst temporarly fix the axis 1 which the shading is bright in the middle, but the light direction is back of the head
        ray[2] *= 0.5

        orth = th.cross(ray, th.tensor([0, 0, 1.0], dtype=th.double).to(device))
        orth2 = th.cross(ray, orth)
        orth = orth / th.norm(orth)
        orth2 = orth2 / th.norm(orth2)

        max_radius = rt_dict['pt_radius']
        big_coords = []
        pt_round = rt_dict['pt_round']

        if pt_round > 1:
            for ti in tqdm.tqdm(range(pt_round), position=1, leave=False, desc="Rendering each perturbed light direction..."):

                tt = ti / (pt_round - 1) * 2 * np.pi * 3
                n = 256 * up_rate
                pray = (orth * np.cos(tt) + orth2 * np.sin(tt)) * (ti / (pt_round - 1)) * max_radius + ray

                mxaxis = max(abs(pray[0]), abs(pray[1]))
                shift = pray / mxaxis * th.arange(n).view(n, 1).to(device)
                coords = depth_grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)
                big_coords.append(coords)
            big_coords = th.cat(big_coords, dim=0)  # [pt_round * n, n, n, 3]
        else:
            n = 256 * up_rate
            pray = ray.clone()
            mxaxis = max(abs(pray[0]), abs(pray[1]))
            shift = pray / mxaxis * th.arange(n).view(n, 1).to(device)
            big_coords = depth_grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)

        output = th.nn.functional.grid_sample(
            th.tensor(np.tile(depth_grid[:, :, 2].view(1, 1, n, n).cpu().numpy(), [n*pt_round, 1, 1, 1])).to(device),
            big_coords[..., :2] / (n - 1) * 2 - 1,
            align_corners=True)

        diff = big_coords[..., 2] - output[:, 0] 
        kk = th.mean((th.min(diff.view(pt_round, -1, n, n), dim=1, keepdim=True)[0] > -0.1) * 1.0, dim=(0, 1)) * 1.0
        out_sd = shadow_mask * kk

        if up_rate > 1 or (out_sd.shape[1:] != (org_h, org_w)):
            out_sd = cv2.resize(out_sd.permute(1, 2, 0).cpu().numpy(), (org_h, org_w), interpolation=cv2.INTER_AREA)[None, ...] # Get the shape 1 x H x W
        else:
            out_sd = out_sd.cpu().numpy()   # Get the shape 1 x H x W
        out_shadow_mask.append(th.tensor(out_sd).to(device))
        out_kk.append(kk[None, ...])
        out_sph_ld.append(th.tensor(image_sph_ld[None, ...]) if image_sph_ld is not None else None)

    return th.cat(out_shadow_mask), th.cat(out_kk), th.cat(out_sph_ld) if image_sph_ld is not None else None

def render_shadow_mask_with_smooth_nopt(sh_light, cam, verts, deca, rt_dict, use_sh_to_ld_region=True, axis_1=False, up_rate=1, device='cuda', org_h=128, org_w=128):
    print("[#] Rendering shadow mask with smooth (no pertubation version)...")
    sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')
    from decalib.utils import util
    from torchvision.transforms import GaussianBlur
    import cv2, tqdm
    
    if verts.shape[0] >= 2:
        tmp = []
        for i in range(1, verts.shape[0]):
            tmp.append(th.allclose(verts[[0]], verts[[i]]))
        assert all(tmp)

    verts = verts[0:1, ...].cuda()  # Save memory by rendering only 1st image
    depth_image_fs, _ = deca['face_scalp'].render.render_depth(verts, up_rate=up_rate)   # Depth : B x 1 x H x W
    B, C, h, w = depth_image_fs.shape

    depth_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='xy')
    depth_grid = np.stack((depth_grid), axis=-1)[None, ...]   # 1 x H x W x 2

    orig = depth_image_fs.detach().clone()
    mask_face = orig != 0
    mask_face = mask_face.float()

    blurred_orig = GaussianBlur(kernel_size=13, sigma=2.0)(orig)
    assert blurred_orig.shape == depth_image_fs.shape

    blurred_mask = GaussianBlur(kernel_size=13, sigma=2.0)(mask_face)
    blurred_mask[blurred_mask == 0] = 1e-10
    blurred_orig = blurred_orig / blurred_mask

    blurred_orig = blurred_orig * (mask_face)

    depth_image_smooth = blurred_orig * 100

    depth_grid = np.concatenate((depth_grid, depth_image_smooth.permute(0, 2, 3, 1)[..., 0:1].cpu().numpy()), axis=-1) # B x H x W x 3

    depth_grid = th.tensor(depth_grid).to(device)   # B x H x W x 3; B should be 1
    assert depth_grid.shape[0] == 1
    depth_grid = depth_grid.squeeze(0)

    out_shadow_mask = []
    out_kk = []
    out_sph_ld = []
    rt_scale = rt_dict['rt_regionG_scale']
    if use_sh_to_ld_region:
        print(f"[#] Approximate the light direction from brightest spot with rt_scale={rt_scale}...")
    else:
        print(f"[#] Approximate the light direction from the mean of SH coffecients [2nd, 3rd and 4th] ...")
    for i in tqdm.tqdm(range(sh_light.shape[0]), position=0, desc="Rendering Shadow Mask on each Sh..."):
        #NOTE: Render the shadow mask from light direction

        shadow_mask = th.clone((depth_image_fs.permute(0, 2, 3, 1)[:, :, :, 0])).to(device)
        if use_sh_to_ld_region:
            ld, misc_dat = sh_to_ld_brightest_region_G(sh=th.tensor(sh_light[[i]]), scale=rt_scale)  # Output shape = (1, 3)
            image_sph_ld = misc_dat[-1]
            ld[:, 2] *= -1
        else:
            ld = sh_to_ld(sh=th.tensor(sh_light[[i]]))  # Output shape = (1, 3)
            image_sph_ld = None
        ld = util.batch_orth_proj(ld[None, ...].cuda(), cam[None, ...].cuda());     # This fn takes pts=Bx3, cam=Bx3
        ld[:, :, 1:] = -ld[:, :, 1:]
        ray = ld.view(3).to(device)
        ray = ray / th.norm(ray)

        if axis_1:
            ray[1] *= -1    # This for jst temporarly fix the axis 1 which the shading is bright in the middle, but the light direction is back of the head
        ray[2] *= 0.5

        n = 256 * up_rate
        pray = ray.clone()
        mxaxis = max(abs(pray[0]), abs(pray[1]))
        shift = pray / mxaxis * th.arange(n).view(n, 1).to(device)
        coords = depth_grid.view(1, n, n, 3) + shift.view(n, 1, 1, 3)

        output = th.nn.functional.grid_sample(
            th.tensor(np.tile(depth_grid[:, :, 2].view(1, 1, n, n).cpu().numpy(), [n, 1, 1, 1])).to(device),
            coords[..., :2] / (n - 1) * 2 - 1,
            align_corners=True)

        diff = coords[..., 2] - output[:, 0] 
        kk = (th.min(diff, dim=0)[0] > -0.1) * 1.0
        out_sd = shadow_mask * kk

        out_sd = out_sd.cpu().numpy()   # Get the shape 1 x H x W
        out_shadow_mask.append(th.tensor(out_sd).to(device))
        out_kk.append(kk[None, ...])
        out_sph_ld.append(th.tensor(image_sph_ld[None, ...]) if image_sph_ld is not None else None)

    return th.cat(out_shadow_mask), th.cat(out_kk), th.cat(out_sph_ld) if image_sph_ld is not None else None

def render_deca(deca_params, idx, n, render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None, repeat=True,
                deca_obj=None):
    '''
    TODO: Adding the rendering with template shape, might need to load mean of camera/tform
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param idx: index of data in batch to render
    :param n: n of repeated tensor (For interpolation)
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        print("[#] No deca_obj, re-init...")
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')
        sys.path.append('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'standard'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    testdata = datasets.TestData([deca_params['raw_image_path'][0]], iscrop=True, face_detector='fan', sample_step=10)
    print(len(testdata), len(deca_params['raw_image_path']))
    if repeat:
        codedict = {'shape':deca_params['shape'][[idx]].repeat(n, 1).to(device).float(),
                    'pose':deca_params['pose'][[idx]].repeat(n, 1).to(device).float(),
                    # 'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':deca_params['exp'][[idx]].repeat(n, 1).to(device).float(),
                    'cam':deca_params['cam'][[idx]].repeat(n, 1).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':deca_params['tform'][[idx]].repeat(n, 1).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].repeat(n, 1, 1).to(device).float(),
                    'images':testdata[idx]['image'].to(device)[None,...].float().repeat(n, 1, 1, 1),
                    'tex':deca_params['albedo'][[idx]].repeat(n, 1).to(device).float(),
                    'detail':deca_params['detail'][[idx]].repeat(n, 1).to(device).float(),
        }
        # print(codedict['pose'])
        # print(codedict['light'])
        # exit()
        original_image = deca_params['raw_image'][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
    else:
        codedict = {'shape':th.tensor(deca_params['shape']).to(device).float(),
                    'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':th.tensor(deca_params['exp']).to(device).float(),
                    'cam':th.tensor(deca_params['cam']).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':th.tensor(deca_params['tform']).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].to(device).float(),
                    'images':th.stack([testdata[i]['image'] for i in range(len(deca_params['raw_image_path']))]).to(device).float(),
                    'tex':th.tensor(deca_params['albedo']).to(device).float(),
                    'detail':(deca_params['detail']).to(device).float(),
        }
        original_image = deca_params['raw_image'].to(device).float() / 255.0
        
    if rotate_normals:
        codedict.update({'R_normals': th.tensor(deca_params['R_normals']).to(device).float()})
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    elif render_mode == 'template_shape':
        use_template = True
        mean_cam = th.tensor(avg_dict['cam'])[None, ...].repeat(n, 1).to(device).float()
        tform = th.tensor(avg_dict['tform'])[None, ...].repeat(n, 1).to(device).reshape(-1, 3, 3).float()
        tform_inv = th.inverse(tform).transpose(1,2)
    else: raise NotImplementedError
    orig_opdict, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict['shape_images']
    return rendered_image, orig_visdict

def render_deca_return_all(deca_params, idx, n, render_mode='shape', 
                useTex=False, extractTex=False, device='cuda', 
                avg_dict=None, rotate_normals=False, use_detail=False,
                deca_mode='only_renderer', mask=None, repeat=True,
                deca_obj=None):
    '''
    TODO: Adding the rendering with template shape, might need to load mean of camera/tform
    # Render the deca face image that used to condition the network
    :param deca_params: dict of deca params = {'light': Bx27, 'shape':BX50, ...}
    :param idx: index of data in batch to render
    :param n: n of repeated tensor (For interpolation)
    :param render_mode: render mode = 'shape', 'template_shape'
    :param useTex: render with texture ***Need the codedict['albedo'] data***
    :param extractTex: for deca texture (set by default of deca decoding pipeline)
    :param device: device for 'cuda' or 'cpu'
    '''
    #import warnings
    #warnings.filterwarnings("ignore")
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cond_utils/DECA/')))
    if deca_obj is None:
        print("[#] No deca_obj, re-init...")
        # sys.path.insert(1, '/home/mint/guided-diffusion/preprocess_scripts/Relighting_preprocessing_tools/DECA/')
        sys.path.insert(0, '/home/mint/guided-diffusion/sample_scripts/cond_utils/DECA/')
        sys.path.append('/home2/mint/Dev/DiFaReli/difareli-faster/sample_scripts/cond_utils/DECA/')
        from decalib import deca
        from decalib.utils.config import cfg as deca_cfg
        deca_cfg.model.use_tex = useTex
        deca_cfg.rasterizer_type = 'standard'
        deca_cfg.model.extract_tex = extractTex
        deca_obj = deca.DECA(config = deca_cfg, device=device, mode=deca_mode, mask=mask)
    else:
        deca_obj = deca_obj
        
    from decalib.datasets import datasets 
    testdata = datasets.TestData([deca_params['raw_image_path'][0]], iscrop=True, face_detector='fan', sample_step=10)
    # print(len(testdata), len(deca_params['raw_image_path']))
    if repeat:
        codedict = {'shape':deca_params['shape'][[idx]].repeat(n, 1).to(device).float(),
                    'pose':deca_params['pose'][[idx]].repeat(n, 1).to(device).float(),
                    # 'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':deca_params['exp'][[idx]].repeat(n, 1).to(device).float(),
                    'cam':deca_params['cam'][[idx]].repeat(n, 1).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':deca_params['tform'][[idx]].repeat(n, 1).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].repeat(n, 1, 1).to(device).float(),
                    'images':testdata[idx]['image'].to(device)[None,...].float().repeat(n, 1, 1, 1),
                    'tex':deca_params['albedo'][[idx]].repeat(n, 1).to(device).float(),
                    'detail':deca_params['detail'][[idx]].repeat(n, 1).to(device).float(),
        }
        # print(codedict['pose'])
        # print(codedict['light'])
        # exit()
        original_image = deca_params['raw_image'][[idx]].to(device).float().repeat(n, 1, 1, 1) / 255.0
    else:
        codedict = {'shape':th.tensor(deca_params['shape']).to(device).float(),
                    'pose':th.tensor(deca_params['pose']).to(device).float(),
                    'exp':th.tensor(deca_params['exp']).to(device).float(),
                    'cam':th.tensor(deca_params['cam']).to(device).float(),
                    'light':th.tensor(deca_params['light']).to(device).reshape(-1, 9, 3).float(),
                    # 'tform':th.tensor(deca_params['tform']).to(device).reshape(-1, 3, 3).float(),
                    'tform':testdata[idx]['tform'][None].to(device).float(),
                    'images':th.stack([testdata[i]['image'] for i in range(len(deca_params['raw_image_path']))]).to(device).float(),
                    'tex':th.tensor(deca_params['albedo']).to(device).float(),
                    'detail':(deca_params['detail']).to(device).float(),
        }
        original_image = deca_params['raw_image'].to(device).float() / 255.0
        
    if rotate_normals:
        codedict.update({'R_normals': th.tensor(deca_params['R_normals']).to(device).float()})
        
    if render_mode == 'shape':
        use_template = False
        mean_cam = None
        tform_inv = th.inverse(codedict['tform']).transpose(1,2)
    elif render_mode == 'template_shape':
        use_template = True
        mean_cam = th.tensor(avg_dict['cam'])[None, ...].repeat(n, 1).to(device).float()
        tform = th.tensor(avg_dict['tform'])[None, ...].repeat(n, 1).to(device).reshape(-1, 3, 3).float()
        tform_inv = th.inverse(tform).transpose(1,2)
    else: raise NotImplementedError
    orig_opdict, orig_visdict = deca_obj.decode(codedict, 
                                  render_orig=True, 
                                  original_image=original_image, 
                                  tform=tform_inv, 
                                  use_template=use_template, 
                                  mean_cam=mean_cam, 
                                  use_detail=use_detail,
                                  rotate_normals=rotate_normals,
                                  )  
    orig_visdict.update(orig_opdict)
    rendered_image = orig_visdict['shape_images']
    return rendered_image, orig_visdict, orig_opdict

def read_params(path):
    params = pd.read_csv(path, header=None, sep=" ", index_col=False, lineterminator='\n')
    params.rename(columns={0:'img_name'}, inplace=True)
    params = params.set_index('img_name').T.to_dict('list')
    return params

def swap_key(params):
    params_s = defaultdict(dict)
    for params_name, v in params.items():
        for img_name, params_value in v.items():
            params_s[img_name][params_name] = np.array(params_value).astype(np.float64)

    return params_s

def normalize(arr, min_val=None, max_val=None, a=-1, b=1):
    '''
    Normalize any vars to [a, b]
    :param a: new minimum value
    :param b: new maximum value
    :param arr: np.array shape=(N, #params_dim) e.g. deca's params_dim = 159
    ref : https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    '''
    if max_val is None and min_val is None:
        max_val = np.max(arr, axis=0)    
        min_val = np.min(arr, axis=0)

    arr_norm = ((b-a) * (arr - min_val) / (max_val - min_val)) + a
    return arr_norm, min_val, max_val

def denormalize(arr_norm, min_val, max_val, a=-1, b=1):
    arr_denorm = (((arr_norm - a) * (max_val - min_val)) / (b - a)) + min_val
    return arr_denorm

def load_params(path, params_key):
    '''
    Load & Return the params
    Input : 
    :params path: path of the pre-computed parameters
    :params params_key: list of parameters name e.g. ['pose', 'light']
    Return :
    :params params_s: the dict-like of {'0.jpg':}
    '''

    params = {}
    for k in params_key:
        for p in glob.glob(f'{path}/*{k}-anno.txt'):
            # Params
            if k in p:
                print(f'Key=> {k} : Filename=>{p}')
                params[k] = read_params(path=p)

    params_s = swap_key(params)

    all_params = []
    for img_name in params_s:
        each_img = []
        for k in params_key:
            each_img.append(params_s[img_name][k])
        all_params.append(np.concatenate(each_img))
    all_params = np.stack(all_params, axis=0)
    return params_s, all_params
    
def get_params_set(set, params_key, path="/data/mint/ffhq_256_with_anno/params/"):
    if set == 'itw':
        # In-the-wild
        sys.path.insert(0, '../../cond_utils/arcface/')
        sys.path.insert(0, '../../cond_utils/arcface/detector/')
        sys.path.insert(0, '../../cond_utils/deca/')
        from cond_utils.arcface import get_arcface_emb
        from cond_utils.deca import get_deca_emb

        itw_path = "../../itw_images/aligned/"
        device = 'cuda:0'
        # ArcFace
        faceemb_itw, emb = get_arcface_emb.get_arcface_emb(img_path=itw_path, device=device)

        # DECA
        deca_itw = get_deca_emb.get_deca_emb(img_path=itw_path, device=device)

        assert deca_itw.keys() == faceemb_itw.keys()
        params_itw = {}
        for img_name in deca_itw.keys():
            params_itw[img_name] = deca_itw[img_name]
            params_itw[img_name].update(faceemb_itw[img_name])
            
        params_set = params_itw
            
    elif set == 'valid' or set == 'train':
        # Load params
        if params_key is None:
            params_key = ['shape', 'pose', 'exp', 'cam', 'light', 'faceemb']

        if set == 'train':
            params_train, params_train_arr = load_params(path=f"{path}/{set}/", params_key=params_key)
            params_set = params_train
        elif set == 'valid':
            params_valid, params_valid_arr = load_params(path=f"{path}/{set}/", params_key=params_key)
            params_set = params_valid
        else:
            raise NotImplementedError

    else: raise NotImplementedError

    return params_set

def preprocess_cond(deca_params, k, cfg):
    if k != 'light':
        return deca_params
    else:
        num_SH = cfg.relighting.num_SH
        params = deca_params[k]
        params = params.reshape(params.shape[0], 9, 3)
        params = params[:, :num_SH, :]
        # params = params.flatten(start_dim=1)
        params = params.reshape(params.shape[0], -1)
        deca_params = params
        return deca_params
    
def write_params(path, params, keys):
    tmp = {}
    for k in keys:
        if th.is_tensor(params[k]):
            tmp[k] = params[k].cpu().numpy()
        else:
            tmp[k] = params[k]
            
    np.save(file=path, arr=tmp)
        