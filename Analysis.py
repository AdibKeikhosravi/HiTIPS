import numpy as np
import cv2
from scipy.ndimage import label
from scipy import ndimage
from PIL import Image
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter

class ImageAnalyzer(object):
    
    def __init__(self,analysisgui):
        self.AnalysisGui = analysisgui

    def neuceli_segmenter(input_img, pixpermic = None):
        
        if self.AnalysisGui.NucDetectMethod.currentText() == "Image Processing":
            
            if self.AnalysisGui.NucCellType.currentText() == "Fibroblasts":
                
                boundary, mask = self.fibroblast_segmenter(input_img, pixpermic = None)
                
            if self.AnalysisGui.NucCellType.currentText() == "MCF10A":
                
                boundary, mask = self.MCF10A_segmenter(input_img, pixpermic = None)
                
            if self.AnalysisGui.NucCellType.currentText() == "HCT116":
                
                boundary, mask = self.HCT116_segmenter(input_img, pixpermic = None)
            
            if self.AnalysisGui.NucCellType.currentText() == "U2OS":
                
                boundary, mask = self.U2OS_segmenter(input_img, pixpermic = None)
            
            if self.AnalysisGui.NucCellType.currentText() == "Mouse Mammary Tumor":
                
                boundary, mask = self.MMT_segmenter(input_img, pixpermic = None)
            
            
            
            
    def fibroblast_segmenter(self, input_img, pixpermic = None):       
            
        uint8_max_val = 255
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ## First blurring round
        median_img = cv2.medianBlur(img_uint8,15)
        #gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
    
    def MCF10A_segmenter(self, input_img, pixpermic = None):       
            
        uint8_max_val = 255
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ## First blurring round
        median_img = cv2.medianBlur(img_uint8,15)
        #gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
    
    def SpotDetector(input_image, AnalysisGui, nuclei_image):
        
        uint8_max_val = 255
    
        ## First blurring round
        median_img = cv2.medianBlur(nuclei_image,11)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/uint8_max_val).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        masked_input = np.multiply(input_image,filled)
        
        sig=3
        if str(AnalysisGui.spotanalysismethod.currentIndex()) == '0':
            
            log_result = ndimage.gaussian_laplace(masked_input, sigma=sig)
            
            if str(AnalysisGui.thresholdmethod.currentIndex()) == '0':
                
                ret_log, thresh_log = cv2.threshold(log_result,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
            elif str(AnalysisGui.thresholdmethod.currentIndex()) == '1':
                
                manual_threshold = AnalysisGui.ThresholdSlider.value()
                ret_log, thresh_log = cv2.threshold(log_result,0,255,cv2.THRESH_BINARY_INV+manual_threshold)
                
            bin_img_log = (1-thresh_log/255).astype('bool')
            spots_img_log = (bin_img_log*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_log = cv2.morphologyEx(spots_img_log, cv2.MORPH_OPEN, kernel)
            final_spots = spot_openned_log
            
        elif str(AnalysisGui.spotanalysismethod.currentIndex()) == '1':
            
            result_gaussian = ndimage.gaussian_filter(masked_input, sigma=sig)
            
            if str(AnalysisGui.thresholdmethod.currentIndex()) == '0':
                
                ret_log, thresh_log = cv2.threshold(result_gaussian,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            elif str(AnalysisGui.thresholdmethod.currentIndex()) == '1':
                
                manual_threshold = AnalysisGui.ThresholdSlider.value()
                
                ret_log, thresh_log = cv2.threshold(result_gaussian,0,255,cv2.THRESH_BINARY_INV+manual_threshold)
            
            bin_img_g = (1-thresh_g/255).astype('bool')
            spots_img_g = (bin_img_g*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_g = cv2.morphologyEx(spots_img_g, cv2.MORPH_OPEN, kernel)
            final_spots = spot_openned_g
        
        ### center of mass calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '0':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            
            bin_img = (final_spots/uint8_max_val).astype('bool')
            ## Binary image filling
            masked_spots = np.multiply(masked_input,bin_img)
            
            spot_locations = ndimage.measurements.center_of_mass(masked_spots, labeled_spots, spot_labels[spot_labels>0])
            
            ###### Brightest spot calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '1':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            bin_img = (final_spots/uint8_max_val).astype('bool')
            masked_spots = np.multiply(masked_input,bin_img)
            spot_locations = peak_local_max(masked_spots, labels=labeled_spots, num_peaks_per_label=1)
        
            ##### Centroid calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '2':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            
            spot_locations = ndimage.measurements.center_of_mass(final_spots, labeled_spots, spot_labels[spot_labelsx>0])
                        
        return spot_locations
    
    def COORDINATES_TO_CIRCLE(coordinates,ImageForSpots):
        
        circles = np.zeros((ImageForSpots.shape), dtype=np.uint8)
        for center_y, center_x in zip(coordinates[:,0], coordinates[:,1]):
                circy, circx = circle_perimeter(center_y, center_x, 7, shape=ImageForSpots.shape)
                circles[circy, circx] = 255


        return circles
    
    def HCT116_segmenter(self, input_img, pixpermic = None):       
            
        uint8_max_val = 255
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ## First blurring round
        median_img = cv2.medianBlur(img_uint8,15)
        #gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
    
    def U2OS_segmenter(self, input_img, pixpermic = None):       
            
        uint8_max_val = 255
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ## First blurring round
        median_img = cv2.medianBlur(img_uint8,15)
        #gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
    
    def MMT_segmenter(self,input_img, pixpermic = None):       
            
        uint8_max_val = 255
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        ## First blurring round
        median_img = cv2.medianBlur(img_uint8,15)
        #gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
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
    