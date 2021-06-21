import numpy as np
import cv2
from scipy.ndimage import label
from scipy import ndimage
from PIL import Image
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
import math
from skimage.filters import median, gaussian, sobel
from skimage.morphology import disk, binary_closing, skeletonize, binary_opening, binary_erosion, white_tophat
from skimage.filters import threshold_li
from skimage.segmentation import watershed, find_boundaries
from skimage.exposure import equalize_adapthist
from skimage.measure import regionprops, regionprops_table
import pandas as pd
from cellpose import models, utils
import tensorflow as tf
from tensorflow.keras import backend as K

class ImageAnalyzer(object):

    def __init__(self,analysisgui, inout_resource_gui):
        self.AnalysisGui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        spot_params_dict =self.INITIALIZE_SPOT_ANALYSIS_PARAMS()
    
    def neuceli_segmenter(self, input_img, pixpermic=None):
        
        if self.AnalysisGui.NucDetectMethod.currentText() == "Int.-based":
            
            first_tresh = self.AnalysisGui.NucSeparationSlider.value()*2.55
            second_thresh = 255-(self.AnalysisGui.NucDetectionSlider.value()*2.55)
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            
            boundary, mask = self.segmenter_function(input_img, cell_size=Cell_Size, 
                                                     first_threshold=first_tresh, second_threshold=second_thresh)
            
        if self.AnalysisGui.NucDetectMethod.currentText() == "Marker Controlled":
            
          
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            max_range = np.sqrt(Cell_Size/3.14)*2/float(pixpermic)
            nuc_detect_sldr = self.AnalysisGui.NucDetectionSlider.value()
            first_tresh = np.ceil((1-(nuc_detect_sldr/100))*max_range).astype(int)
            
            second_thresh = self.AnalysisGui.NucSeparationSlider.value()
            
            
            boundary, mask = self.watershed_scikit(input_img, cell_size=Cell_Size, 
                                                     first_threshold=first_tresh, second_threshold=second_thresh)

        if self.AnalysisGui.NucDetectMethod.currentText() == "CellPose-GPU":
            
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=True, cell_dia=cell_diameter)
            
        if self.AnalysisGui.NucDetectMethod.currentText() == "CellPose-CPU":
            
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            cell_diameter = np.sqrt(Cell_Size/(float(pixpermic)*float(pixpermic)))*2/3.14
            
            boundary, mask = self.cellpose_segmenter(input_img, use_GPU=False, cell_dia=cell_diameter)
            
        if self.AnalysisGui.NucDetectMethod.currentText() == "DeepCell":
            
            Cell_Size = self.AnalysisGui.NucleiAreaSlider.value()
            cell_diameter = Cell_Size/100
            
            boundary, mask = self.deepcell_segmenter(input_img, cell_dia=cell_diameter)
              
        return boundary, mask   


    def deepcell_segmenter(self, input_img, cell_dia=None):
        
        from deepcell.applications import NuclearSegmentation
        app = NuclearSegmentation()
        im = np.expand_dims(input_img, axis=-1)
        im = np.expand_dims(im, axis=0)
        
        masks1 = app.predict(im, image_mpp=cell_dia)
        masks = np.squeeze(masks1)
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)
        boundary_img = (255*boundary).astype('uint8')
        resized_bound = boundary_img
        filled1 = ndimage.binary_fill_holes(resized_bound)
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')

        return boundary, mask
        
    def cellpose_segmenter(self, input_img, use_GPU, cell_dia=None):
        
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        model = models.Cellpose(gpu=use_GPU, model_type='nuclei')
        masks, flows, styles, diams = model.eval(img_uint8, diameter=cell_dia, flow_threshold=None)
	
        boundary = find_boundaries(masks, connectivity=1, mode='thick', background=0)
#         boundary_img = (255*boundary).astype('uint8')
        boundary_img = (255*boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]).astype('uint8')
        resized_bound = cv2.resize(boundary_img,(input_img.shape[1],input_img.shape[0]))
#         resized_bound = boundary_img
        filled1 = ndimage.binary_fill_holes(resized_bound)
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')

        return boundary, mask

    def segmenter_function(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
    
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        ## First blurring round
        if (cell_size %2)==0:
            cell_size = cell_size + 1
        median_img = cv2.medianBlur(img_uint8,cell_size)
        gaussian_blurred = cv2.GaussianBlur(median_img,(5,5),0)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(gaussian_blurred,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/255).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        filled_int= (filled*255).astype('uint8')
        ## Gray2RGB to feed the watershed algorithm
        img_rgb  = cv2.cvtColor(img_uint8,cv2.COLOR_GRAY2RGB)
        boundary = img_rgb
        boundary = boundary - img_rgb
        ## Distance trasform and thresholing to set the watershed markers
        dt = cv2.distanceTransform(filled.astype(np.uint8), 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, first_threshold, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl = lbl.astype(np.int32)
        ## First round of Watershed transform
        cv2.watershed(img_rgb, lbl)
        ## Correcting image boundaries
        boundary[lbl == -1] = [255,255,255]
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
        ret2, thresh2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        dt = cv2.distanceTransform(255-thresh2, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, second_threshold, 255, cv2.THRESH_BINARY)
        lbl, ncc = label(dt)
        lbl = lbl * (255 / (ncc + 1))
        lbl = lbl.astype(np.int32)
        cv2.watershed(img_rgb, lbl)
        ########
        boundary = img_rgb
        boundary = boundary - img_rgb

        boundary[lbl == -1] = [255,255,255]
        boundary_img = boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]
        bound_gray = cv2.cvtColor(boundary_img,cv2.COLOR_BGR2GRAY)
        resized_bound = cv2.resize(bound_gray,(input_img.shape[1],input_img.shape[0]))

        kernel = np.ones((3,3),np.uint8)
        boundary = cv2.dilate(resized_bound,kernel,iterations = 1)
        filled1 = ndimage.binary_fill_holes(boundary)
        fin= 255*filled1-boundary
        mask = ndimage.binary_fill_holes(fin)
        mask = (255*mask).astype(np.uint8)

        return boundary, mask
            
    def watershed_scikit(self, input_img, cell_size=None, first_threshold=None, second_threshold=None):
        
        img_uint8 = cv2.copyMakeBorder(input_img,5,5,5,5,cv2.BORDER_CONSTANT,value=0)
        
        med_scikit = median(img_uint8, disk(1))
        thresh = threshold_li(med_scikit)
        binary = med_scikit > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_blurred = gaussian(filled, 1)
        filled_int= (filled_blurred*255).astype('uint8')
        
# #         edge_sobel = sobel(img_uint8)
# #         enhanced = 50*edge_sobel/edge_sobel.max() + img_uint8
# #         enhanced.astype('uint8')
# #         med_scikit = median(img_uint8, disk(5))
        thresh = threshold_li(filled_int)
        binary = filled_int > thresh
        filled = ndimage.binary_fill_holes(binary)
        filled_int = binary_opening(filled, disk(5))
        filled_int = ndimage.binary_fill_holes(filled_int)
#         filled_blurred = gaussian(openeed, 3)
        
#         thresh = threshold_li(filled_int)
#         binary = filled_int > thresh
        #binary = binary_erosion(filled_int, disk(5))
        distance = ndimage.distance_transform_edt(filled_int)
        binary1 = distance > first_threshold
        distance1 = ndimage.distance_transform_edt(binary1)
        binary2 = distance1 > second_threshold

        labeled_spots, num_features = label(binary2)
        spot_labels = np.unique(labeled_spots)    

        spot_locations = ndimage.measurements.center_of_mass(binary2, labeled_spots, spot_labels[spot_labels>0])

        mask = np.zeros(distance.shape, dtype=bool)
        if spot_locations:
            mask[np.ceil(np.array(spot_locations)[:,0]).astype(int), np.ceil(np.array(spot_locations)[:,1]).astype(int)] = True
        markers, _ = ndimage.label(mask)
        labels = watershed(-distance, markers, mask=binary, compactness=0.5, watershed_line=True)
        boundary = find_boundaries(labels, connectivity=1, mode='thick', background=0)
        boundary_img = (255*boundary[3:boundary.shape[0]-3,3:boundary.shape[1]-3]).astype('uint8')
        resized_bound = cv2.resize(boundary_img,(input_img.shape[1],input_img.shape[0]))
        filled1 = ndimage.binary_fill_holes(resized_bound)
        mask= (255*filled1).astype('uint8')-resized_bound
        boundary= resized_bound.astype('uint8')
        
        return boundary, mask
    
    def max_z_project(self, image_stack):
        
        z_imglist=[]
        
        for index, row in image_stack.iterrows():
            im = Image.open(row['ImageName'])
            z_imglist.append( np.asarray(im))
        z_stack = np.stack(z_imglist, axis=2)
        max_project = z_stack.max(axis=2)
        
        return max_project
    
   
    def SpotDetector(self, input_image, AnalysisGui, nuclei_image, spot_channel):
        
        self.UPDATE_SPOT_ANALYSIS_PARAMS()
        if AnalysisGui.spotchannelselect.currentText()=='All':
            
            params_to_pass= self.spot_params_dict['Ch1']
        else:
            params_to_pass= self.spot_params_dict[spot_channel]
        
        
        uint8_max_val = 255
    
        ## First blurring round
        median_img = cv2.medianBlur(nuclei_image,11)
        ## Threhsolding and Binarizing
        ret, thresh = cv2.threshold(median_img,0,uint8_max_val,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        bin_img = (1-thresh/uint8_max_val).astype('bool')
        ## Binary image filling
        filled = ndimage.binary_fill_holes(bin_img)
        struct = ndimage.generate_binary_structure(2, 2)
        filled = ndimage.binary_dilation(filled, structure=struct).astype(filled.dtype)
        filled = ndimage.binary_dilation(filled, structure=struct).astype(filled.dtype)
        #### this part is for removing bright junk in the image################
        labeled_nuc, num_features_nuc = label(filled)
        props = regionprops_table(labeled_nuc, input_image, properties=('label', 'area', 'max_intensity', 'mean_intensity'))
        props_df = pd.DataFrame(props)
        mean_intensity_ave=props_df['mean_intensity'].mean()
        max_intensity_max=props_df['max_intensity'].max()
        for ind, row in props_df.iterrows():
            
            if row['mean_intensity'] > 2*mean_intensity_ave:
                
                input_image[labeled_nuc==row['label']]=0
        input_image[input_image>max_intensity_max]=0       
        input_image = cv2.normalize(input_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
########################################
        sig=params_to_pass[3]
        if str(params_to_pass[0]) == '0':
            
            log_result = ndimage.gaussian_laplace(input_image, sigma=sig)
            if str(params_to_pass[1]) == '0':
                

                ret_log, thresh_log = cv2.threshold(log_result,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_log = (1-thresh_log/255).astype('bool')

            if str(params_to_pass[1]) == '1':
                
                manual_threshold = np.ceil(params_to_pass[2]*2.55).astype(int)
                thresh_log = log_result > manual_threshold
                bin_img_log = thresh_log
#             bin_img_log = (1-thresh_log/255).astype('bool')
            spots_img_log = (bin_img_log*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_log = cv2.morphologyEx(spots_img_log, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_log,filled)
            
        if str(params_to_pass[0]) == '1':
            
            result_gaussian = ndimage.gaussian_filter(input_image, sigma=sig)
            
            if str(params_to_pass[1]) == '0':
                
                ret_log, thresh_log = cv2.threshold(result_gaussian,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                bin_img_g = (1-thresh_log/255).astype('bool')
                
            if str(params_to_pass[1]) == '1':
                
                manual_threshold = np.ceil(params_to_pass[2]*2.55).astype(int)
                
                thresh_log = result_gaussian > manual_threshold
                bin_img_g = thresh_log
            
            spots_img_g = (bin_img_g*255).astype('uint8')
            kernel = np.ones((3,3), np.uint8)
            spot_openned_g = cv2.morphologyEx(spots_img_g, cv2.MORPH_OPEN, kernel)
            final_spots = np.multiply(spot_openned_g,filled)

        ### center of mass calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '0':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            
            bin_img = (final_spots/uint8_max_val).astype('bool')
            ## Binary image filling
            masked_spots = np.multiply(input_image,bin_img)
            
            spot_locations = ndimage.measurements.center_of_mass(masked_spots, labeled_spots, spot_labels[spot_labels>0])
            
            ###### Brightest spot calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '1':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            bin_img = (final_spots/uint8_max_val).astype('bool')
            masked_spots = np.multiply(input_image,bin_img)
            spot_locations = peak_local_max(masked_spots, labels=labeled_spots, num_peaks_per_label=1)
        
            ##### Centroid calculation
        if str(AnalysisGui.SpotLocationCbox.currentIndex()) == '2':
            
            labeled_spots, num_features = label(final_spots)
            spot_labels = np.unique(labeled_spots)
            
            spot_locations = ndimage.measurements.center_of_mass(final_spots, labeled_spots, spot_labels[spot_labels>0])
                        
        return spot_locations, final_spots
    
    def COORDINATES_TO_CIRCLE(self, coordinates,ImageForSpots):
        
        circles = np.zeros((ImageForSpots.shape), dtype=np.uint8)
        
        if coordinates.any():
            
            for center_y, center_x in zip(coordinates[:,0], coordinates[:,1]):
                    circy, circx = circle_perimeter(center_y, center_x, 7, shape=ImageForSpots.shape)
                    circles[circy, circx] = 255

        return circles
    
    def SPOTS_TO_BOUNDARY(self, final_spots):
        
        labeled_spots, num_features = label(final_spots)
        boundary = find_boundaries(labeled_spots, connectivity=1, mode='thick', background=0)
        spot_boundary = (255*boundary).astype('uint8')
        
        return spot_boundary
    
    
    def INITIALIZE_SPOT_ANALYSIS_PARAMS(self):
        
        self.spot_params_dict={
            
            "Ch1": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch2": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch3": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch4": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch5": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
            }
        return self.spot_params_dict
    
    def UPDATE_SPOT_ANALYSIS_PARAMS(self):
        
        if self.AnalysisGui.spotchannelselect.currentText()=='All':
            
            self.spot_params_dict={
            
            "Ch1": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch2": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch3": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch4": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int),
            "Ch5": np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
            }
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch1':
            
            self.spot_params_dict["Ch1"] =    np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
            
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch2':
            
            self.spot_params_dict["Ch2"] =    np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch3':
            
            self.spot_params_dict["Ch3"] =    np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
            
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch4':
            
            self.spot_params_dict["Ch4"] =    np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
            
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch5':
            
            self.spot_params_dict["Ch5"] =    np.array([self.AnalysisGui.spotanalysismethod.currentIndex(),self.AnalysisGui.thresholdmethod.currentIndex(),
                    self.AnalysisGui.ThresholdSlider.value(), self.AnalysisGui.SensitivitySpinBox.value(),
                            self.AnalysisGui.SpotPerChSpinBox.value()],dtype=int)
    
    def UPDATE_SPOT_ANALYSIS_GUI_PARAMS(self):
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch1':
            
            self.AnalysisGui.spotanalysismethod.setCurrentIndex(np.array(self.spot_params_dict["Ch1"][0]).astype(int))
            self.AnalysisGui.thresholdmethod.setCurrentIndex(np.array(self.spot_params_dict["Ch1"][1]).astype(int))
            self.AnalysisGui.ThresholdSlider.setValue(np.array(self.spot_params_dict["Ch1"][2]).astype(int))
            self.AnalysisGui.SensitivitySpinBox.setValue(np.array(self.spot_params_dict["Ch1"][3]).astype(int))
            self.AnalysisGui.SpotPerChSpinBox.setValue(np.array(self.spot_params_dict["Ch1"][4]).astype(int))
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch2':
            
            self.AnalysisGui.spotanalysismethod.setCurrentIndex(np.array(self.spot_params_dict["Ch2"][0]).astype(int))
            self.AnalysisGui.thresholdmethod.setCurrentIndex(np.array(self.spot_params_dict["Ch2"][1]).astype(int))
            self.AnalysisGui.ThresholdSlider.setValue(np.array(self.spot_params_dict["Ch2"][2]).astype(int))
            self.AnalysisGui.SensitivitySpinBox.setValue(np.array(self.spot_params_dict["Ch2"][3]).astype(int))
            self.AnalysisGui.SpotPerChSpinBox.setValue(np.array(self.spot_params_dict["Ch2"][4]).astype(int))
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch3':
            
            self.AnalysisGui.spotanalysismethod.setCurrentIndex(np.array(self.spot_params_dict["Ch3"][0]).astype(int))
            self.AnalysisGui.thresholdmethod.setCurrentIndex(np.array(self.spot_params_dict["Ch3"][1]).astype(int))
            self.AnalysisGui.ThresholdSlider.setValue(np.array(self.spot_params_dict["Ch3"][2]).astype(int))
            self.AnalysisGui.SensitivitySpinBox.setValue(np.array(self.spot_params_dict["Ch3"][3]).astype(int))
            self.AnalysisGui.SpotPerChSpinBox.setValue(np.array(self.spot_params_dict["Ch3"][4]).astype(int))
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch4':
            
            self.AnalysisGui.spotanalysismethod.setCurrentIndex(np.array(self.spot_params_dict["Ch4"][0]).astype(int))
            self.AnalysisGui.thresholdmethod.setCurrentIndex(np.array(self.spot_params_dict["Ch4"][1]).astype(int))
            self.AnalysisGui.ThresholdSlider.setValue(np.array(self.spot_params_dict["Ch4"][2]).astype(int))
            self.AnalysisGui.SensitivitySpinBox.setValue(np.array(self.spot_params_dict["Ch4"][3]).astype(int))
            self.AnalysisGui.SpotPerChSpinBox.setValue(np.array(self.spot_params_dict["Ch4"][4]).astype(int))
        
        if self.AnalysisGui.spotchannelselect.currentText()=='Ch5':
            
            self.AnalysisGui.spotanalysismethod.setCurrentIndex(np.array(self.spot_params_dict["Ch5"][0]).astype(int))
            self.AnalysisGui.thresholdmethod.setCurrentIndex(np.array(self.spot_params_dict["Ch5"][1]).astype(int))
            self.AnalysisGui.ThresholdSlider.setValue(np.array(self.spot_params_dict["Ch5"][2]).astype(int))
            self.AnalysisGui.SensitivitySpinBox.setValue(np.array(self.spot_params_dict["Ch5"][3]).astype(int))
            self.AnalysisGui.SpotPerChSpinBox.setValue(np.array(self.spot_params_dict["Ch5"][4]).astype(int))
        
