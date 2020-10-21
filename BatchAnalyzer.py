import numpy as np
import cv2
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image, ImageQt
from Analysis import ImageAnalyzer
from scipy.ndimage import label
from joblib import Parallel, delayed
WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class BatchAnalysis(object):
    def __init__(self,analysisgui):
        self.AnalysisGui = analysisgui
    def ON_APPLYBUTTON(self, Meta_Data_df, displaygui, inout_resource_gui, ImDisplay, PlateGrid):
        
        ch1_spot_df, ch2_spot_df, ch3_spot_df, ch4_spot_df, ch5_spot_df, cell_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        df_cols = [ "Column", "Row", "Time Point", "Field Index", "ZSlice", "Channel", "Action Index"]
        
        displaygui.setEnabled(False)
        columns = np.unique(np.asarray(Meta_Data_df['Column'], dtype=int))
        rows = np.unique(np.asarray(Meta_Data_df['Row'], dtype=int))
        fovs = np.unique(np.asarray(Meta_Data_df['FieldIndex'], dtype=int))
        timepoints = np.unique(np.asarray(Meta_Data_df['TimePoint'], dtype=int))
        actionindices = np.unique(np.asarray(Meta_Data_df['ActionIndex'], dtype=int))

        for col in columns:
            for row in rows:
                for fov in fovs:
                    for t in timepoints:
                         for ai in actionindices:
                        
                            df_checker = Meta_Data_df.loc[(Meta_Data_df['Column'] == str(col)) & (Meta_Data_df['Row'] == str(row)) & 
                                                    (Meta_Data_df['FieldIndex'] == str(fov)) & (Meta_Data_df['TimePoint'] == str(t))]
                            
                                    
                            ImageForNucMask = self.IMG_FOR_NUC_MASK(df_checker)
                            
                            if ImageForNucMask.ndim ==2:

                                nuc_bndry, nuc_mask = ImageAnalyzer.neuceli_segmenter(ImageForNucMask)
                                labeled_nuc, number_nuc = label(nuc_mask)
                                nuc_labels = np.unique(labeled_nuc)

                                nuc_centroid_locations = ndimage.measurements.center_of_mass(nuc_mask, labeled_nuc, 
                                                                                             nuc_labels[nuc_labels>0])
                                data = { "Column": [col]*number_nuc, "Row": [row]*number_nuc, 
                                         "Time Point": [t]*number_nuc, "Field Index": [fov]*number_nuc,
                                         "ZSlice": ["max project"]*number_nuc}
                                df = pd.DataFrame(data)
                                regions = regionprops(labeled_nuc, ImageForNucMask)
                                props = regionprops_table(labeled_nuc, ImageForNucMask, properties=(
                                                        'centroid', 'orientation', 'major_axis_length', 'minor_axis_length',
                                                        'area', 'label' , 'max_intensity', 'min_intensity', 'mean_intensity',
                                                        'orientation', 'perimeter'))
                                
                                cell_df1 = pd.DataFrame(props)
                                cell_df = pd.concat([df,cell_df1], axis=1)
                                
                            else:
                                
                                nuc_bndry, nuc_mask = self.Z_STACK_NUC_SEGMENTER(ImageForNucMask)
                                label_nuc_stack = self.Z_STACK_NUC_LABLER(ImageForLabel)

                            ch1_xyz, ch1_xyz_3D, ch2_xyz, ch2_xyz_3D, ch3_xyz, ch3_xyz_3D, ch4_xyz, ch4_xyz_3D, ch5_xyz, ch5_xyz_3D = self.IMAGE_FOR_SPOT_DETECTION( df_checker, ImageForNucMask)

                            if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:

                                if self.AnalysisGui.SpotMaxZProject.isChecked() == True:
                                    #with pd.ExcelWriter('output.xlsx', engine="openpyxl", mode='a') as writer:  
                                            
                                        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:
                                            if ch1_xyz!=[]:
                                                ch1_xyz_round = np.round(np.asarray(ch1_xyz)).astype('int')
                                                ch1_spot_nuc_labels = labeled_nuc[ch1_xyz_round[:,0], ch1_xyz_round[:,1]]
                                                ch1_num_spots = ch1_xyz.__len__()
                                                added_columns = ["cell index","ch1_x_location","ch1_y_location","ch1_z_location"]
                                                df_cols.append(added_columns)

                                                data = { "Column": [col]*ch1_num_spots, "Row": [row]*ch1_num_spots, 
                                                         "Time Point": [t]*ch1_num_spots, "Field Index": [fov]*ch1_num_spots,
                                                         "ZSlice": ["max project"]*ch1_num_spots, "Channel": [1]*ch1_num_spots,
                                                         "Action Index": [ai]*ch1_num_spots, "cell index": ch1_spot_nuc_labels,
                                                         "ch1_x_location": ch1_xyz[:,0], "ch1_y_location": ch1_xyz[:,1],
                                                         "ch1_z_location": ch1_xyz[:,2]}
                                                df = pd.DataFrame(data)
                                                ch1_spot_df=pd.concat([ch1_spot_df,df],ignore_index=True)
                                                

                                        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
                                            if ch2_xyz!=[]:
                                                ch2_xyz_round = np.round(np.asarray(ch2_xyz)).astype('int')
                                                ch2_spot_nuc_labels = labeled_nuc[ch2_xyz_round[:,0], ch2_xyz_round[:,1]]
                                                ch2_num_spots = ch2_xyz.__len__()
                                                added_columns = ["cell index","ch2_x_location","ch2_y_location","ch2_z_location"]
                                                df_cols.append(added_columns)

                                                data = { "Column": [col]*ch2_num_spots, "Row": [row]*ch2_num_spots, 
                                                         "Time Point": [t]*ch2_num_spots, "Field Index": [fov]*ch2_num_spots,
                                                         "ZSlice": ["max project"]*ch2_num_spots, "Channel": [2]*ch2_num_spots,
                                                         "Action Index": [ai]*ch2_num_spots, "cell index": ch2_spot_nuc_labels,
                                                         "ch2_x_location": ch2_xyz[:,0], "ch2_y_location": ch2_xyz[:,1],
                                                         "ch2_z_location": ch2_xyz[:,2]}
                                                df = pd.DataFrame(data)
                                                ch2_spot_df=pd.concat([ch2_spot_df,df],ignore_index=True)
                                               
                                        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
                                            if ch3_xyz!=[]:

                                                ch3_xyz_round = np.round(np.asarray(ch3_xyz)).astype('int')
                                                ch3_spot_nuc_labels = labeled_nuc[ch3_xyz_round[:,0], ch3_xyz_round[:,1]]
                                                ch3_num_spots = ch3_xyz.__len__()
                                                added_columns = ["cell index","ch3_x_location","ch3_y_location","ch3_z_location"]
                                                df_cols.append(added_columns)
                                                data = { "Column": [col]*ch3_num_spots, "Row": [row]*ch3_num_spots, 
                                                         "Time Point": [t]*ch3_num_spots, "Field Index": [fov]*ch3_num_spots,
                                                         "ZSlice": ["max project"]*ch3_num_spots, "Channel": [3]*ch3_num_spots,
                                                         "Action Index": [ai]*ch3_num_spots, "cell index": ch3_spot_nuc_labels,
                                                         "ch3_x_location": ch3_xyz[:,0], "ch3_y_location": ch3_xyz[:,1],
                                                         "ch3_z_location": ch3_xyz[:,2]}

                                                df = pd.DataFrame(data)
                                                ch3_spot_df=pd.concat([ch3_spot_df,df],ignore_index=True)
                                                
                                        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
                                            if ch4_xyz!=[]:

                                                ch4_xyz_round = np.round(np.asarray(ch4_xyz)).astype('int')
                                                ch4_spot_nuc_labels = labeled_nuc[ch4_xyz_round[:,0], ch4_xyz_round[:,1]]
                                                ch4_num_spots = ch4_xyz.__len__()
                                                added_columns = ["cell index","ch4_x_location","ch4_y_location","ch4_z_location"]
                                                df_cols.append(added_columns)

                                                data = { "Column": [col]*ch4_num_spots, "Row": [row]*ch4_num_spots, 
                                                         "Time Point": [t]*ch4_num_spots, "Field Index": [fov]*ch4_num_spots,
                                                         "ZSlice": ["max project"]*ch4_num_spots, "Channel": [4]*ch4_num_spots,
                                                         "Action Index": [ai]*ch4_num_spots, "cell index": ch4_spot_nuc_labels,
                                                         "ch4_x_location": ch4_xyz[:,0], "ch4_y_location": ch4_xyz[:,1],
                                                         "ch4_z_location": ch4_xyz[:,2]}

                                                df = pd.DataFrame(data)
                                                ch4_spot_df=pd.concat([ch4_spot_df,df],ignore_index=True)
                                                

                                        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
                                            if ch5_xyz!=[]:

                                                ch5_xyz_round = np.round(np.asarray(ch5_xyz)).astype('int')
                                                ch5_spot_nuc_labels = labeled_nuc[ch5_xyz_round[:,0], ch5_xyz_round[:,1]]
                                                ch5_num_spots = ch5_xyz.__len__()
                                                added_columns = ["cell index","ch5_x_location","ch5_y_location","ch5_z_location"]
                                                df_cols.append(added_columns)

                                                data = { "Column": [col]*ch5_num_spots, "Row": [row]*ch5_num_spots, 
                                                         "Time Point": [t]*ch5_num_spots, "Field Index": [fov]*ch5_num_spots,
                                                         "ZSlice": ["max project"]*ch5_num_spots, "Channel": [5]*ch5_num_spots,
                                                         "Action Index": [ai]*ch5_num_spots, "cell index": ch5_spot_nuc_labels,
                                                         "ch5_x_location": ch5_xyz[:,0], "ch5_y_location": ch5_xyz[:,1],
                                                         "ch5_z_location": ch5_xyz[:,2]}

                                                df = pd.DataFrame(data)
                                                ch4_spot_df=pd.concat([ch4_spot_df,df],ignore_index=True)
                                            
                                    
        if ch1_spot_df.empty == False:
            
            ch1_spot_df.to_excel('ch1_spot_df.xlsx')
            
        if ch2_spot_df.empty == False:
            
            ch2_spot_df.to_excel('ch2_spot_df.xlsx')    
            
        if ch3_spot_df.empty == False:
            
            ch3_spot_df.to_excel('ch3_spot_df.xlsx')   
            
        if ch4_spot_df.empty == False:
            
            ch4_spot_df.to_excel('ch4_spot_df.xlsx')    
            
        if ch5_spot_df.empty == False:
            
            ch5_spot_df.to_excel('ch5_spot_df.xlsx')    
            
            
        
                                                    
                            
                            
                            
                                
                                
                        
                            


                            











    def IMG_FOR_NUC_MASK(self, df_checker):
        
        if df_checker.empty == False:

            if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:

                maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
                imgformask = df_checker.loc[(df_checker['Channel'] == maskchannel)]
                loadedimg_formask = ImageAnalyzer.max_z_project(imgformask)
                ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            else:
                z_imglist=[]
                maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
                imgformask = df_checker.loc[(df_checker['Channel'] == maskchannel)]

                for index, row in imgformask.iterrows():
                    im = mpimg.imread(row['ImageName'])
                    im_uint8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    z_imglist.append( np.asarray(im_uint8))
                ImageForNucMask = np.stack(z_imglist, axis=2)
        
        return ImageForNucMask

    def Z_STACK_NUC_SEGMENTER(self, ImageForNucMask):
        
        nuc_bndry_list, nuc_mask_list = [], []
        w, h, z_plane = ImageForNucMask.shape

        for z in range(z_plane):

            single_z_img = ImageForNucMask[:,:,z]
            nuc_bndry_single, nuc_mask_single = ImageAnalyzer.neuceli_segmenter(single_z_img)

            nuc_bndry_list.append( np.asarray(nuc_bndry_single))
            nuc_mask_list.append( np.asarray(nuc_mask_single))
        nuc_bndry = np.stack(nuc_bndry_list, axis=2)
        nuc_mask = np.stack(nuc_mask_list, axis=2)
    
        return nuc_bndry, nuc_mask
    
    def Z_STACK_NUC_LABLER(self, ImageForLabel):
        
        label_nuc_list = []
        w, h, z_plane = ImageForLabel.shape

        for z in range(z_plane):

            single_z_img = ImageForLabel[:,:,z]
            labeled_nuc, number_nuc = label(single_z_img)

            label_nuc_list.append( np.asarray(labeled_nuc))
            
        label_nuc_stack = np.stack(label_nuc_list, axis=2)
    
        return label_nuc_stack
    
    
    def IMAGE_FOR_SPOT_DETECTION(self, df_checker, ImageForNucMask):
        
        ch1_xyz, ch2_xyz, ch3_xyz, ch4_xyz, ch5_xyz = [],[],[],[],[]
        ch1_xyz_3D, ch2_xyz_3D, ch3_xyz_3D, ch4_xyz_3D, ch5_xyz_3D = [],[],[],[],[]
        
        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:

            imgforspot = df_checker.loc[(df_checker['Channel'] == '1')]
            ch1_xyz, ch1_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
                
        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
            
            imgforspot = df_checker.loc[(df_checker['Channel'] == '2')]
            ch2_xyz, ch2_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
                
        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
            
            imgforspot = df_checker.loc[(df_checker['Channel'] == '3')]
            ch3_xyz, ch3_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
                    
        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
             
            imgforspot = df_checker.loc[(df_checker['Channel'] == '4')]
            ch4_xyz, ch4_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
        
        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
             
            imgforspot = df_checker.loc[(df_checker['Channel'] == '5')]
            ch5_xyz, ch5_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
            
        return ch1_xyz, ch1_xyz_3D, ch2_xyz, ch2_xyz_3D, ch3_xyz, ch3_xyz_3D, ch4_xyz, ch4_xyz_3D, ch5_xyz, ch5_xyz_3D

    
    def XYZ_SPOT_COORDINATES(self, images_pd_df, ImageForNucMask):
        
        z_imglist = []
        coordinates_list = []
        print(images_pd_df)
        for index, row in images_pd_df.iterrows():
            im = mpimg.imread(row['ImageName'])
            _z_coordinates1 = np.asarray(row['ZSlice']).astype('float')
            im_uint8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            coordinates = ImageAnalyzer.SpotDetector(im_uint8, self.AnalysisGui, ImageForNucMask)
            
            if coordinates.__len__()>0:
                _z_coordinates = np.ones((coordinates.__len__(),1), dtype='float')*_z_coordinates1
            else:
                coordinates=np.ones((0,1), dtype='float')
                _z_coordinates=np.ones((0,1), dtype='float')
                
            xyz_3d_coordinates = np.append(np.asarray(coordinates).astype('float'), _z_coordinates, 1)
            coordinates_list.append(xyz_3d_coordinates)
            z_imglist.append( np.asarray(im_uint8))
            
        coordinates_stack = np.stack(coordinates_list, axis=2)
        image_stack = np.stack(z_imglist, axis=2)
        max_project = image_stack.max(axis=2)
        coordinates_max_project = ImageAnalyzer.SpotDetector(im_uint8, self.AnalysisGui, ImageForNucMask)
        if coordinates_max_project.__len__()>0:
            coordinates_max_project_round = np.round(np.asarray(coordinates_max_project)).astype('int')
            spots_z_slices = np.argmax(image_stack[coordinates_max_project_round[:,0],coordinates_max_project_round[:,1],:], axis=1)
            spots_z_coordinates = np.zeros((spots_z_slices.__len__(),1), dtype='float')
            for i in range(spots_z_slices.__len__()):

                spots_z_coordinates[i] = np.asarray(images_pd_df.loc[images_pd_df['ZSlice']== str(spots_z_slices[i]+1)]
                                                 ['Z_coordinate'].iloc[0]).astype('float')

            xyz_coordinates = np.append(np.asarray(coordinates_max_project).astype('float'), spots_z_coordinates, 1)
        else:
            
            xyz_coordinates = []
            
            
        return xyz_coordinates, coordinates_stack