import numpy as np
import cv2
from scipy.ndimage import label
from scipy import ndimage
from PIL import Image

class ImageAnalyzer(object):

    def neuceli_segmenter(input_img, pixpermic = None):
        uint8_max_val = 255
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ## First blurring round
        median_img = cv2.medianBlur(img_uint8,31)
        gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(gaussian_blurred,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/uint8_max_val).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        filled_int= (filled*uint8_max_val).astype('uint8')
        ## Gray2RGB to feed the watershed algorithm
        img_rgb  = cv2.cvtColor(img_uint8,cv2.COLOR_GRAY2RGB)
        boundary = img_rgb
        boundary = boundary - img_rgb
        ## Distance trasform and thresholing to set the watershed markers
        dt = cv2.distanceTransform(filled.astype(np.uint8), 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * uint8_max_val).astype(np.uint8)
        _, dt = cv2.threshold(dt, 40, uint8_max_val, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (uint8_max_val / (ncc + 1))
        lbl = lbl.astype(np.int32)
        ## First round of Watershed transform
        cv2.watershed(img_rgb, lbl)
        ## Correcting image boundaries
        boundary[lbl == -1] = [uint8_max_val,uint8_max_val,uint8_max_val]
        boundary[0,:] = 0
        boundary[-1,:] = 0
        boundary[:,0] = 0
        boundary[:, -1] = 0
        b_gray = cv2.cvtColor(boundary,cv2.COLOR_BGR2GRAY)
        diff = filled_int-b_gray

        kernel = np.ones((11,11), np.uint8)
        first_pass = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)

        ## Second round of marker generation and watershed 
        kernel = np.ones((5,5),np.uint8)
        aa = first_pass.astype('uint8')
        erosion = cv2.erode(aa,kernel,iterations = 1)
        kernel = np.ones((11,11), np.uint8)
        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        blur = cv2.GaussianBlur(opening,(11,11),50)
        ret2, thresh2 = cv2.threshold(blur,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        dt = cv2.distanceTransform(uint8_max_val-thresh2, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * uint8_max_val).astype(np.uint8)
        _, dt = cv2.threshold(dt, 80, uint8_max_val, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (uint8_max_val / (ncc + 1))
        lbl = lbl.astype(np.int32)
        cv2.watershed(img_rgb, lbl)
        ########
        boundary = img_rgb
        boundary = boundary - img_rgb

        boundary[lbl == -1] = [uint8_max_val,uint8_max_val,uint8_max_val]
        boundary_img = boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]
        bound_gray = cv2.cvtColor(boundary_img,cv2.COLOR_BGR2GRAY)
        resized_bound = cv2.resize(bound_gray,(input_img.shape[1],input_img.shape[0]))

        kernel = np.ones((3,3),np.uint8)
        boundary = cv2.dilate(resized_bound,kernel,iterations = 1)
        filled1 = ndimage.binary_fill_holes(boundary)
        fin= uint8_max_val*filled1-boundary
        mask = ndimage.binary_fill_holes(fin)
        mask = (uint8_max_val*mask).astype(np.uint8)

        return boundary, mask
    
    def max_z_project( image_stack):
        
        z_imglist=[]
        
        for index, row in image_stack.iterrows():
            im = Image.open(row['ImageName'])
            z_imglist.append( np.asarray(im))
        z_stack = np.stack(z_imglist, axis=2)
        max_project = z_stack.max(axis=2)
        
        return max_project
    
    def LOG_spotDetector(input_image, sig=None):
        
        sig=3
        log_result = ndimage.gaussian_laplace(input_image, sigma=sig)
        ret2, thresh2 = cv2.threshold(log_result,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh2/255).astype('bool')
        spots_img= (bin_img*255).astype('uint8')
        
        return spots_img

