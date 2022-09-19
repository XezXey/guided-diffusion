#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2

#%%
def rgb_to_sepia(img):
    r = img[..., [0]].copy()
    g = img[..., [1]].copy()
    b = img[..., [2]].copy()
    r_s = (r * 0.393) + (g * 0.769) + (b * 0.189)
    g_s = (r * 0.349) + (g * 0.686) + (b * 0.168)
    b_s = (r * 0.272) + (g * 0.534) + (b * 0.131)

    img_sepia = np.concatenate((r_s, g_s, b_s), axis=2)
    img_sepia = np.clip(img_sepia, 0, 255).astype(np.int)

    return img_sepia

def rgb_sw_chn(img, ch='rgb'):
    assert len(ch) == 3
    r, g, b = 0, 1 ,2
    ch_n = []
    for c in ch:
        if c == 'r':
            ch_n.append(r)
        elif c == 'g':
            ch_n.append(g)
        elif c == 'b':
            ch_n.append(b)
        else:
            raise NotImplementedError

    img_ = img[..., ch_n].copy()
    return img_

def rgb_to_hsv(img):
    img_ = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img_

def rgb_to_hls(img):
    img_ = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return img_

def rgb_to_ycrcb(img):
    img_ = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return img_

def rgb_to_luv(img):
    img_ = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    return img_

def img_recolor(img, out_c):
    if out_c == 'sepia':
        img_ = rgb_to_sepia(img)
    elif out_c == 'hsv':
        img_ = rgb_to_hsv(img)
    elif out_c == 'hls':
        img_ = rgb_to_hls(img)
    elif out_c == 'ycrcb':
        img_ = rgb_to_ycrcb(img)
    elif out_c == 'luv':
        img_ = rgb_to_luv(img)
    elif out_c in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr']:
        img_ = rgb_sw_chn(img, ch=out_c)
    else: raise NotImplementedError

    return img_


#%%
img = cv2.imread('./6.png')[..., ::-1]
for ch in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr', 'hsv', 'hls', 'ycrcb', 'luv', 'sepia']:
    img_ = img_recolor(img.copy(), ch)
    print(np.max(img_), np.min(img_))
    plt.imshow(img_)
    plt.title(ch)
    plt.show()
#%%

img = cv2.imread('./6.png')[..., ::-1]
plt.imshow(img)
plt.show()

img_sepia = rgb_to_sepia(img)
img_sepia = np.clip(img_sepia, 0, 255).astype(np.int)
plt.imshow(img_sepia)
plt.show()

for ch in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr']:
    img_ = rgb_sw_chn(img.copy(), ch)
    plt.imshow(img_)
    plt.title(ch)
    plt.show()

img_hsv = rgb_to_hsv(img)
plt.imshow(img_hsv)
plt.show()

# %%
for ch in ['rgb', 'rbg', 'brg', 'bgr', 'grb', 'gbr', 'hsv', 'hls', 'ycrcb', 'luv', 'sepia']:
    img_ = img_recolor(img.copy(), ch)
    print(img_.shape)
    plt.imshow(img_)
    plt.title(ch)
    plt.show()