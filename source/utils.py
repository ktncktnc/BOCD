import os
import numpy as np
from skimage import io
from scipy.ndimage import zoom


def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    
    # crop if necesary
    I = I[:s[0],:s[1]]
    si = I.shape
    
    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])
    
    return np.pad(I,((0,p0),(0,p1)),'edge')
    

def read_sentinel_img(path, NORMALISE_IMGS = True):
    im_name = os.listdir(path)[0][:-7]
    path_img_name = os.path.join(path, im_name)

    r = io.imread(os.path.join(path_img_name, "B04.tif"))
    g = io.imread(os.path.join(path_img_name, "B03.tif"))
    b = io.imread(os.path.join(path_img_name, "B02.tif"))
    
    I = np.stack((r,g,b),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_4(path, NORMALISE_IMGS = True):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    path_img_name = os.path.join(path, im_name)

    r = io.imread(os.path.join(path_img_name, "B04.tif"))
    g = io.imread(os.path.join(path_img_name, "B03.tif"))
    b = io.imread(os.path.join(path_img_name, "B02.tif"))
    nir = io.imread(os.path.join(path_img_name, "B08.tif"))
    
    I = np.stack((r,g,b,nir),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_leq20(path, NORMALISE_IMGS = True):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]
    path_img_name = os.path.join(path, im_name)
    
    r = io.imread(os.path.join(path_img_name, "B04.tif"))
    s = r.shape
    g = io.imread(os.path.join(path_img_name, "B03.tif"))
    b = io.imread(os.path.join(path_img_name, "B02.tif"))
    nir = io.imread(os.path.join(path_img_name, "B08.tif"))
    
    ir1 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B05.tif")),2),s)
    ir2 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B06.tif")),2),s)
    ir3 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B07.tif")),2),s)
    nir2 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B8A.tif")),2),s)
    swir2 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B11.tif")),2),s)
    swir3 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B12.tif")),2),s)
    
    I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_leq60(path, NORMALISE_IMGS = True):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]
    path_img_name = os.path.join(path, im_name)
    
    r = io.imread(os.path.join(path_img_name, "B04.tif"))
    s = r.shape
    g = io.imread(os.path.join(path_img_name, "B03.tif"))
    b = io.imread(os.path.join(path_img_name, "B02.tif"))
    nir = io.imread(os.path.join(path_img_name, "B08.tif"))
    
    ir1 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B05.tif")),2),s)
    ir2 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B06.tif")),2),s)
    ir3 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B07.tif")),2),s)
    nir2 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B8A.tif")),2),s)
    swir2 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B11.tif")),2),s)
    swir3 = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B12.tif")),2),s)
    
    uv = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B01.tif")),6),s)
    wv = adjust_shape(zoom(io.imread(os.path.join(path_img_name,"B09.tif"),6),s)
    swirc = adjust_shape(zoom(io.imread(os.path.join(path_img_name, "B10.tif")),6),s)
    
    I = np.stack((r,g,b,nir,ir1,ir2,ir3,nir2,swir2,swir3,uv,wv,swirc),axis=2).astype('float')
    
    if NORMALISE_IMGS:
        I = (I - I.mean()) / I.std()

    return I

def read_sentinel_img_trio(path, img_type = 0, NORMALISE_IMGS = True):
    """Read cropped Sentinel-2 image pair and change map."""
#     read images
    if img_type == 0:
        I1 = read_sentinel_img(path + '/imgs_1/', NORMALISE_IMGS)
        I2 = read_sentinel_img(path + '/imgs_2/', NORMALISE_IMGS)

    elif img_type == 1:
        I1 = read_sentinel_img_4(path + '/imgs_1/', NORMALISE_IMGS)
        I2 = read_sentinel_img_4(path + '/imgs_2/', NORMALISE_IMGS)

    elif img_type == 2:
        I1 = read_sentinel_img_leq20(path + '/imgs_1/', NORMALISE_IMGS)
        I2 = read_sentinel_img_leq20(path + '/imgs_2/', NORMALISE_IMGS)

    elif img_type == 3:
        I1 = read_sentinel_img_leq60(path + '/imgs_1/', NORMALISE_IMGS)
        I2 = read_sentinel_img_leq60(path + '/imgs_2/', NORMALISE_IMGS)
        
    cm = io.imread(path + '/cm/cm.png', as_gray=True) != 0
    
    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2,((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0,0)),'edge')
    
    
    return I1, I2, cm



def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
#     out = np.swapaxes(I,1,2)
#     out = np.swapaxes(out,0,1)
#     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)