from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import numpy as np
import time
import pandas as pd
import matplotlib.image as mpimg
import cv2
from PIL import Image
from skimage import exposure
from PIL import Image, ImageQt
import qimage2ndarray
from Analysis import ImageAnalyzer
from AnalysisGUI import analyzer
from scipy.ndimage import label
from skimage.color import label2rgb

class imagedisplayer(analyzer,QWidget):
    METADATA_DATAFRAME = pd.DataFrame()
    imgchannels = pd.DataFrame()
    grid_data = np.zeros(5, dtype = int)
    def __init__(self,analysisgui,centralwidget):
        #super(self, analyzer).__init__(centralwidget)
        
        self._zoom = 0
        self._empty = True
        
        self.ch1_hist_max = 255
        self.ch1_hist_min = 0
        self.ch2_hist_max = 255
        self.ch2_hist_min = 0
        self.ch3_hist_max = 255
        self.ch3_hist_min = 0
        self.ch4_hist_max = 255
        self.ch4_hist_min = 0
        self.AnalysisGui = analysisgui
        
    def display_initializer(self, out_df, displaygui, IO_GUI):
            
            displaygui.setEnabled(True)
            self.METADATA_DATAFRAME = out_df
            
            displaybtn = IO_GUI.DisplayCheckBox
            if displaybtn.isChecked() == True:
            # Image scroller and spinbox initialization
                numoffiles = np.asarray(out_df['Column'], dtype=int).__len__()
            
            
                # Histogram Max Min initialization for slider and spinbox

                displaygui.MaxHistSlider.setMaximum(255)
                displaygui.MaxHistSlider.setMinimum(0)
                displaygui.MaxHistSlider.setValue(255)

                displaygui.MinHistSlider.setMaximum(255)
                displaygui.MinHistSlider.setMinimum(0)
                displaygui.MinHistSlider.setValue(0)
 
                
                
            else:
                pass

    
    ##### Image histogram controller funcions            
      
    def MAX_HIST_SLIDER_UPDATE(self, displaygui):
            
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            
            if self.MaxSlider_ind <= self.MinSlider_ind:
                
                displaygui.MinHistSlider.setValue(self.MaxSlider_ind)
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
           
    def MIN_HIST_SLIDER_UPDATE(self, displaygui):
            
            self.MinSlider_ind = displaygui.MinHistSlider.value()
            self.MaxSlider_ind = displaygui.MaxHistSlider.value()
            
            if self.MinSlider_ind >= self.MaxSlider_ind:
                
                displaygui.MaxHistSlider.setValue(self.MinSlider_ind)
                
            
            self.READ_IMAGE(displaygui, self.imgchannels)
            
    def GET_IMAGE_NAME(self,displaygui):
            
            
            self.imgchannels = self.METADATA_DATAFRAME.loc[
                                    (self.METADATA_DATAFRAME['Column'] == str(self.grid_data[0])) & 
                                    (self.METADATA_DATAFRAME['Row'] == str(self.grid_data[1])) & 
                                    (self.METADATA_DATAFRAME['TimePoint'] == str(self.grid_data[2])) & 
                                    (self.METADATA_DATAFRAME['FieldIndex'] == str(self.grid_data[3])) & 
                                    (self.METADATA_DATAFRAME['ZSlice'] == str(self.grid_data[4]))
                                        ]
            self.READ_IMAGE(displaygui, self.imgchannels)            
    
    def READ_IMAGE(self, displaygui, image_channels):
            
            self.height = 0
            self.width = 0

            if 'ch1_img' in locals():
                del ch1_img
            if 'ch2_img' in locals():
                del ch2_img
            if 'ch3_img' in locals():
                del ch3_img
            if 'ch4_img' in locals():
                del ch4_img
            if 'All_Channels' in locals():
                del All_Channels
            
                
            if displaygui.Ch1CheckBox.isChecked() == True:
                #print(self.imgchannels.loc[self.imgchannels['Channel']=='1']['ImageName'].iloc[0])
                ch1_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='1']['ImageName'].iloc[0])
                ch1_img = (ch1_img/256).astype('uint8')
                self.CH1_img = ch1_img
                self.height, self.width = np.shape(ch1_img)
                
            if displaygui.Ch2CheckBox.isChecked() == True:

                ch2_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='2']['ImageName'].iloc[0])
                ch2_img = (ch2_img/256).astype('uint8')
                self.height, self.width = np.shape(ch2_img)

            if displaygui.Ch3CheckBox.isChecked() == True:

                ch3_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='3']['ImageName'].iloc[0])
                ch3_img = (ch3_img/256).astype('uint8')
                self.height, self.width = np.shape(ch3_img)
                
            if displaygui.Ch4CheckBox.isChecked() == True:

                ch4_img = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']=='4']['ImageName'].iloc[0])
                ch4_img = (ch4_img/256).astype('uint8')
                self.height, self.width = np.shape(ch4_img)
            
            if self.height or self.width:
                
                self.RGB_channels = np.zeros((self.height, self.width, 3))
                if 'ch2_img' in locals():
                    self.RGB_channels[:,:,0] = ch2_img
                if 'ch3_img' in locals():
                    self.RGB_channels[:,:,1] = ch3_img
                if 'ch4_img' in locals():
                    self.RGB_channels[:,:,2] = ch4_img
                
                self.ADJUST_IMAGE_CONTRAST(displaygui, self.CH1_img, self.RGB_channels )
               
            
    def ADJUST_IMAGE_CONTRAST(self, displaygui, CH1 , RGB_CHANNELS):
            
            RGB_Channels = RGB_CHANNELS.astype(np.uint8)
            self.lower = displaygui.MinHistSlider.value()
            self.upper = displaygui.MaxHistSlider.value()

            if displaygui.HistChannel.currentText() == "Ch 1":
                
                self.ch1_hist_max = self.upper
                self.ch1_hist_min = self.lower
                
            if 'RGB_CHANNELS' in locals():  
            
                
                if displaygui.HistChannel.currentText() == "Ch 2":
                    
                    self.ch2_hist_max = self.upper
                    self.ch2_hist_min = self.lower
                    
                if displaygui.HistChannel.currentText() == "Ch 3":
                    
                    self.ch3_hist_max = self.upper
                    self.ch3_hist_min = self.lower
                
                if displaygui.HistChannel.currentText() == "Ch 4":
                    
                    self.ch4_hist_max = self.upper
                    self.ch4_hist_min = self.lower
                    
                CH1 = self.ON_ADJUST_INTENSITY(CH1, self.ch1_hist_min, self.ch1_hist_max)
                RGB_Channels[:,:,0] = self.ON_ADJUST_INTENSITY(RGB_Channels[:,:,0], self.ch2_hist_min, self.ch2_hist_max)
                RGB_Channels[:,:,1] = self.ON_ADJUST_INTENSITY(RGB_Channels[:,:,1], self.ch3_hist_min, self.ch3_hist_max)
                RGB_Channels[:,:,2] = self.ON_ADJUST_INTENSITY(RGB_Channels[:,:,2], self.ch4_hist_min, self.ch4_hist_max)
            self.MERGEIAMGES(displaygui, CH1, RGB_Channels)
            
    def MERGEIAMGES(self,displaygui, CH1, RGB_Channels):
        
            if displaygui.Ch1CheckBox.isChecked() == True:
                ch1_rgb = np.stack((CH1,)*3, axis=-1)
            else:
                ch1_rgb = np.zeros(RGB_Channels.shape, dtype = np.uint8)
            All_Channels = cv2.addWeighted(ch1_rgb, 1, RGB_Channels, 1, 0)
            height, width, ch = np.shape(All_Channels)
            totalBytes = All_Channels.nbytes
            #print(self.AnalysisGui.NucleiChannel.currentIndex().dtype)
            if displaygui.NuclMaskCheckBox.isChecked() == True:
                
                self.input_image = self.IMAGE_TO_BE_MASKED()
                bound, filled_res = ImageAnalyzer.neuceli_segmenter(self.input_image)
                #cv2.imwrite('mask_saved.jpg',bound)
                if displaygui.NucPreviewMethod.currentText() == "Boundary":
                    
                    All_Channels[bound != 0] = [255,0,0]
                    
                if displaygui.NucPreviewMethod.currentText() == "Area":
                
                    labeled_array, num_features = label(filled_res)
                    rgblabel = label2rgb(labeled_array,alpha=0.1, bg_label = 0)
                    rgblabel = cv2.normalize(rgblabel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    image_input_stack = np.stack((self.input_image,)*3, axis=-1)
                    All_Channels = cv2.addWeighted(rgblabel,0.2, ch1_rgb, 1, 0)
                    ##############
            if displaygui.SpotsCheckBox.isChecked() == True:
                
                self.input_image = self.IMAGE_TO_BE_MASKED()
                ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img = self.IMAGE_FOR_SPOT_DETECTION(self.input_image)

                if displaygui.SpotPreviewMethod.currentText() == "Dots":
                    
                    if ch1_spots_img!=[]:
                    
                        All_Channels[ch1_spots_img != 0] = [255,255,255]
                        
                    if ch2_spots_img!=[]:
                    
                        All_Channels[ch2_spots_img != 0] = [255,0,0]
                        
                    if ch3_spots_img!=[]:
                    
                        All_Channels[ch3_spots_img != 0] = [0,255,0]
                        
                    if ch4_spots_img!=[]:
                    
                        All_Channels[ch4_spots_img != 0] = [0,0,255]
                    
                    
                if displaygui.NucPreviewMethod.currentText() == "Cross":
                    
                    pass
                

            self.SHOWIMAGE(displaygui, All_Channels, width, height, totalBytes)
            
    def SHOWIMAGE(self, displaygui, img, width, height, totalBytes):
            
                        
            displaygui.viewer.setPhoto(QtGui.QPixmap.fromImage(qimage2ndarray.array2qimage(img)))
            
    def IMAGE_TO_BE_MASKED(self):
        
        if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:
            maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
            self.imgformask = self.METADATA_DATAFRAME.loc[
                                    (self.METADATA_DATAFRAME['Column'] == str(self.grid_data[0])) & 
                                    (self.METADATA_DATAFRAME['Row'] == str(self.grid_data[1])) & 
                                    (self.METADATA_DATAFRAME['TimePoint'] == str(self.grid_data[2])) & 
                                    (self.METADATA_DATAFRAME['FieldIndex'] == str(self.grid_data[3])) &  
                                    (self.METADATA_DATAFRAME['Channel'] == maskchannel)
                                    ]
            loadedimg_formask = ImageAnalyzer.max_z_project(self.imgformask)
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            
            maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
            
            loadedimg_formask = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== maskchannel]['ImageName'].iloc[0])
            
            ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return ImageForNucMask
       
    
    def IMAGE_FOR_SPOT_DETECTION(self, nuclei_image):
        
        ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img = [],[],[],[]
        
        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['Channel'] == '1')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch1_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                
            else:

                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '1']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch1_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
        

        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['Channel'] == '2')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch2_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                
            else:
                
                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '2']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch2_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
       
                
        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(self.grid_data[3])) & 
                                        (self.METADATA_DATAFRAME['Channel'] == '3')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch3_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
            else:

                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '3']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch3_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
        
                
        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                self.imgforspot = self.METADATA_DATAFRAME.loc[
                                        (self.METADATA_DATAFRAME['Column'] == str(self.grid_data[0])) & 
                                        (self.METADATA_DATAFRAME['Row'] == str(self.grid_data[1])) & 
                                        (self.METADATA_DATAFRAME['TimePoint'] == str(self.grid_data[2])) & 
                                        (self.METADATA_DATAFRAME['FieldIndex'] == str(self.grid_data[3])) &  
                                        (self.METADATA_DATAFRAME['Channel'] == '4')
                                        ]
                loadedimg_forspot = ImageAnalyzer.max_z_project(self.imgforspot)
                ImageForSpots = cv2.normalize(loadedimg_forspot, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch4_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
                
            else:

                loadedimg_forspots = mpimg.imread(self.imgchannels.loc[self.imgchannels['Channel']== '4']['ImageName'].iloc[0])
                ImageForSpots = cv2.normalize(loadedimg_forspots, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                coordinates = ImageAnalyzer.SpotDetector(ImageForSpots, self.AnalysisGui, nuclei_image)
                ch1_spots_img = ImageAnalyzer.COORDINATES_TO_CIRCLE(np.round(coordinates).astype('int'),ImageForSpots)
            
        return ch1_spots_img, ch2_spots_img, ch3_spots_img, ch4_spots_img


    def ON_ADJUST_INTENSITY(self, input_img, min_range, max_range):
        eplsion = 0.005
        mid_img = np.zeros((input_img.shape),dtype='uint16')
        input_img[input_img < min_range] = 0
        mid_img = input_img.astype('uint16')
        
        scale_factor = 255/(max_range + eplsion)
        mid_img = np.round(mid_img * scale_factor).astype('uint16')
        
        mid_img[mid_img > 255] = 255
        #output_img = cv2.normalize(input_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        output_img = mid_img.astype('uint8')
        return output_img