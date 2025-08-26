import numpy as np
import pyshtools
import torch

# module load OpenBLAS/0.3.20-GCC-11.3.0

def get_shcoeff(image, Lmax=100):
    """
    @param image: image in HWC @param 1max: maximum of sh
    """
    output_coeff = []
    for c_id in range(image.shape[-1]):
        # Create a SHGrid object from the image
        grid = pyshtools.SHGrid.from_array(image[:,:,c_id], grid='GLQ')
        # Compute the spherical harmonic coefficients
        coeffs = grid.expand(normalization='4pi', csphase=1, lmax_calc=Lmax)
        coeffs = coeffs.to_array()
        output_coeff.append(coeffs[None])
    
    output_coeff = np.concatenate(output_coeff,axis=0)
    return output_coeff

def unfold_sh_coeff(flatted_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient from 3x9 (e.g., level=2) to 3 x 2 x 3 x C matrix (when C = max_sh_level+1)
    #  array format [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    
    Output: 
    sh_coeff[i, 1, D, D] is [-m, 0)
    sh_coeff[i, 0, D, D] is [0, +m]
    e.g., 
    1) unfolded[0][0] is channel = 0 and +m side
        working on sh[0] (same channel)
    [
        [sh[0], 0, 0,],
        [sh[2], sh[3], 0],
        [sh[6], sh[7], sh[8]]
    ]
    
    2) unfolded[0][1] is channel = 0 and -m side
        working on sh[0] (same channel)
    [
        [0, 0, 0,],
        [0, sh[1], 0],
        [0, sh[4], sh[5]]
    ]

    """
    sh_coeff = np.zeros((3, 2, max_sh_level+1, max_sh_level+1))
    for c in range(3):  # Iterate each channel
        coeff = 0   # Each coeff
        for j in range(max_sh_level+1): # Iterate each level    [0 to max_sh_level]
            # if j == 1: continue
            for k in range(j, 0, -1):   # From [-(max_sh_level) to -1]; inclusive
                sh_coeff[c, 1, j, k] = flatted_coeff[c, coeff]
                coeff +=1
            for k in range(0, j+1, 1):  # From [0 to max_sh_level]; inclusive
                sh_coeff[c, 0, j, k] = flatted_coeff[c, coeff]
                coeff += 1
    return sh_coeff

def flatten_sh_coeff(sh_coeff, max_sh_level=2):
    """
    flatten spherical harmonics coefficient to 3xC matrix
    """
    flatted_coeff = np.zeros((3, (max_sh_level+1) ** 2))
    # we will put into array in the format of 
    # [0_0, 1_-1, 1_0, 1_1, 2_-2, 2_-1, 2_0, 2_1, 2_2]
    # where first number is the order and the second number is the position in order
    for i in range(3):
        c = 0
        for j in range(max_sh_level+1):
            # if j == 1: continue
            for k in range(j, 0, -1):
                flatted_coeff[i, c] = sh_coeff[i, 1, j, k]
                c +=1
            for k in range(j+1):
                flatted_coeff[i, c] = sh_coeff[i, 0, j, k]
                c += 1
    return flatted_coeff

def compute_background(sh, lmax=2, image_width=512):
    # Generate random spherical harmonic coefficients
    loaded_coeff = sh
    
    output_image = []
    for ch in (range(3)):
        coeffs = loaded_coeff[ch]
            
        # Create SHCoeffs class object from the coefficients
        sh_coeffs = pyshtools.SHCoeffs.from_array(coeffs, lmax=lmax, normalization='4pi', csphase=1)

        # Create a grid of latitudes and longitudes
        theta = np.linspace(np.pi / 2, -np.pi / 2, image_width)
        phi = np.linspace(0, np.pi * 2, 2*image_width)


        lat, lon = np.meshgrid(theta, phi, indexing='ij')

        # Evaluate the spherical harmonics on the grid
        grid_data = sh_coeffs.expand(grid="GLQ", lat=lat, lon=lon, lmax_calc=lmax, degrees=False)
        output_image.append(grid_data[...,None])    

    output_image = np.concatenate(output_image,axis=-1)
    return output_image

def sample_from_sh(shcoeff, lmax, theta, phi):
    """
    Sample envmap from sh 
    """
    assert shcoeff.shape[0] == 3 # make sure that it a 3 channel input
    output = []
    for ch in (range(3)):
        coeffs = pyshtools.SHCoeffs.from_array(shcoeff[ch], lmax=lmax, normalization='4pi', csphase=1)
        image = coeffs.expand(grid="GLQ", lat=theta, lon=phi, lmax_calc=lmax, degrees=False)
        # image = coeffs.expand(grid="DH2", lat=theta, lon=phi, lmax_calc=lmax, degrees=False)
        output.append(image[...,None])
    output = np.concatenate(output, axis=-1)
    return output

def get_ideal_normal_ball(size, flip_x=True):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen    
    (we flip X to match sobel operator)
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x

    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0

    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    normal_map = normal_map.numpy()
    mask = mask.numpy()
    return normal_map, mask

def get_ideal_normal_ball_z_up(size):
    """
    Generate front size of normal ball that has z up
    """
    y = np.linspace(-1, 1, size)
    z = np.linspace(1, -1, size)
    y, z = np.meshgrid(y, z)
    
    # avoid negative value
    x2 = 1 - y**2 - z**2
    mask = x2 >= 0

    # get real x value
    x = np.sqrt(np.clip(x2,0,1))    

    x = x * mask
    y = y * mask
    z = z * mask
    # set x outside mask to be 1
    x = x + (1 - mask)
    normal_map = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)

    return normal_map, mask

def cartesian_to_spherical(vectors):
    """
    Converts unit vectors to spherical coordinates (theta, phi).

    Parameters:
    vectors (numpy.ndarray): Input array of shape (..., 3), representing unit vectors.

    Returns:
    tuple: A tuple containing two arrays:
        - theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
        - phi (numpy.ndarray): Array of phi values in the range [0, 2*pi].
    """
    # Ensure input is a numpy array
    vectors = np.asarray(vectors)

    # Validate shape
    if vectors.shape[-1] != 3:
        raise ValueError("Input must have shape (..., 3).")

    # Extract components of the vectors
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]

    # Calculate theta (latitude angle)
    theta = np.arcsin(z)  # arcsin gives range [-pi/2, pi/2]
    # theta = np.arccos(z) - np.pi/2  # arcsin gives range [-pi/2, pi/2]

    # Calculate phi (longitude angle)
    phi = np.arctan2(y, x)  # atan2 accounts for correct quadrant
    phi = (phi + 2 * np.pi) % (2 * np.pi)  # Normalize phi to range [0, 2*pi]

    return theta, phi
    
def spherical_to_cartesian(theta, phi):
    """
    Converts spherical coordinates (theta, phi) to unit vectors.

    Parameters:
    theta (numpy.ndarray): Array of theta values in the range [-pi/2, pi/2].
    phi (numpy.ndarray): Array of phi values in the range [0, 2*pi].

    Returns:
    numpy.ndarray: Output array of shape (..., 3), representing unit vectors.
    """
    # Ensure inputs are numpy arrays
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Calculate components of the unit vectors
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    # Stack components into output array
    vectors = np.stack([x, y, z], axis=-1)

    return vectors

def get_uniform_rays_dense_top(H, W, num_rays):
    """
    random uniform rays for each pixel 
    Parameters:
    - H (int): height of the image
    - W (int): width of the image
    - num_rays (int): number of rays to sample
    Returns:
    - np.ndarray: ray direction in [H,W, num_rays, 3]
    """
    # generate random rays
    theta = np.random.uniform(0, np.pi/2, (H, W, num_rays)) # we only sample half sphere
    phi = np.random.uniform(0, 2*np.pi, (H, W, num_rays))
    rays = spherical_to_cartesian(theta, phi)   
    return rays

def get_uniform_rays(H, W, num_rays):
    """
    random phi angle (azimuth) and random Z then we compute X and Y value
    AXIS connvetion: y-right z-up
    """
    phi = np.random.uniform(0, 2 * np.pi, (H, W, num_rays))  # Azimuth angle
    z = np.random.uniform(0, 1, (H, W, num_rays))  # Sample z in (0,1) to ensure positive hemisphere

    r = np.sqrt(1 - z**2)  # Radius for the x-y plane to keep unit vector constraint

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return np.stack([x, y, z], axis=-1)  # Shape: (H, W, num_rays, 3)

def get_uniform_rays_reject_sample(H, W, num_rays):
    """
    random_rejection_method 
    # https://blog.thomaspoulet.fr/uniform-sampling-on-unit-hemisphere/
    """
    finished_sample = False 
    sample_count = 0
    expected_num_rays = H * W * num_rays
    while not finished_sample:
        if sample_count > 100:
            raise Exception("There is something wrong with random ray process. please try again")
        # just random sample x and y
        x = np.random.uniform(-1, 1, (expected_num_rays * 10))
        y = np.random.uniform(-1, 1, (expected_num_rays * 10))
        # filter out z that negative
        z2 =  1 - x**2 - y**2
        mask = z2 >= 0
        x = x[mask]
        y = y[mask]
        z2 = z2[mask]
        if x.shape[0]  < expected_num_rays:
            sample_count += 1
            continue 
        x = x[:expected_num_rays].reshape((H,W,num_rays))
        y = y[:expected_num_rays].reshape((H,W,num_rays))
        z2 = z2[:expected_num_rays].reshape((H,W,num_rays))
        z = np.sqrt(z2)
        finished_sample = True 
        break
    rays = np.stack([x, y, z], axis=-1)
    return rays



def get_uniform_rays_normalize_method(H, W, num_rays):
    # normalize method is not good
    x = np.random.uniform(-1, 1, (H, W, num_rays)) # we only sample half sphere
    y = np.random.uniform(-1, 1, (H, W, num_rays))
    z = np.random.uniform(0, 1, (H, W, num_rays))
    rays = np.stack([x, y, z], axis=-1)
    # normalize to unit vector
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays



def get_uniform_rays_reject_sampling(H, W, num_rays):
    """
    random rays that more uniformly by random x and y, then  we compute z 
    AXIS connvetion: y-right z-up
    """
    x,y = np.random.uniform(-1, 1, (H, W, num_rays * 100))

    return np.stack([x, y, z], axis=-1)  # Shape: (H, W, num_rays, 3)


def get_rotation_matrix_from_vectors_single(a, b):
    """
    Find the rotation matrix that aligns vector a to vector b
    Parameters:
    - a (np.ndarray): vector a in [3]
    - b (np.ndarray): vector b in [3]
    Returns:
    - np.ndarray: rotation matrix in [3,3]
    """
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    # avoid parallel vectors
    if s == 0:
        if c > 0:
            return np.eye(3)
        else:
            return -np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / (s ** 2)
    return R    

def apply_integrate_conv(shcoeff, Lmax):
    # apply integrate on diffuse surface 
    # @see https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
    assert shcoeff.shape[0] == 3 and shcoeff.shape[1] == 2
    
    A = np.array([
        np.pi, # 0
        2*np.pi / 3, # 1
        np.pi / 4, # 2
    ])
    for j in range(3):  # Iterate each 
        # check if it still access
        if j < shcoeff.shape[2]:
            shcoeff[:,:,j] = A[j] * shcoeff[:,:,j]
    return shcoeff

def apply_integrate_conv_anyLmax(shcoeff, Lmax):
    # apply integrate on diffuse surface 
    # @see https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
    '''
    shcoeff: unfolded_sh_coeff in shape [3, 2, Lmax + 1, Lmax + 1]
    '''
    assert shcoeff.shape[0] == 3 and shcoeff.shape[1] == 2
    import scipy
    A = []
    for l in range(Lmax + 1):
        if l == 1:
            out = 2 * np.pi / 3
        elif l > 1 and l % 2 == 1:
            out = 0
        elif l % 2 == 0:
            fact = scipy.special.factorial(l) / ((2 ** l) * (scipy.special.factorial(l / 2) ** 2))
            out = 2 * np.pi * ((((-1) ** ((l / 2) - 1))) / ((l + 2) * (l - 1))) * fact
        A.append(out)
        
    A = np.array(A)
    for j in range(Lmax + 1):  # Iterate each 
        # check if it still access
        if j < shcoeff.shape[2]:
            shcoeff[:,:,j] = A[j] * shcoeff[:,:,j]
    return shcoeff

# TODO: need unit testing
def from_x_left_to_z_up(point):
    """
    Convert from ControlNet x-left, y-up, z-forward to x-forward, y-right z-up
    """
    assert point.shape[-1] == 3 # only support catesian coordinate
    rotation_matrix = np.array([
        [0., 0., 1.], # new x-forward  coming  from z-foward
        [-1., 0., 0.], # new y-right coming from x-left
        [0., 1., 0.], # new z-up comfing from y-up
    ])
    # convert to torch to multiply in last 2 dimension.
    rotation_matrix = torch.from_numpy(rotation_matrix).float()
    point =  torch.from_numpy(point)[...,None].float()
    new_point = rotation_matrix @ point 
    new_point = new_point[...,0].numpy() # shape [H,W,3]
    return new_point

def genSurfaceNormals(n):
    """
    convention is 
    x is left->right [-1, 1]
    y is up->down [1, -1] 
    z is back->front [0, 1]
    """
    x = torch.linspace(-1, 1, n)
    y = torch.linspace(1, -1, n)
    y, x = torch.meshgrid(y, x)

    z = (1 - x ** 2 - y ** 2)   # x^2 + y^2 + z^2 = 1
    z[z < 0] = 0    # Outside hemisphere = 0, including outside circle
    alpha = z != 0 
    z = torch.sqrt(z)
    return torch.stack([x, y, z], 0), alpha