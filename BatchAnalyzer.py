import numpy as np
import cv2
import math
import os
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image, ImageQt
from scipy.ndimage import label
import multiprocessing
from joblib import Parallel, delayed
WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class BatchAnalysis(object):
    
    def __init__(self,analysisgui, image_analyzer):
        
        self.AnalysisGui = analysisgui
        self.ImageAnalyzer = image_analyzer
    
        
    def ON_APPLYBUTTON(self, Meta_Data_df, displaygui, inout_resource_gui, ImDisplay, PlateGrid):
        
        if inout_resource_gui.NumCPUsSpinBox.value()==0:
            jobs_number=1
        else:
            jobs_number=inout_resource_gui.NumCPUsSpinBox.value()
        Parallel(n_jobs=jobs_number)(delayed(self.RUN_BATCH_ANALYZER( Meta_Data_df, displaygui, inout_resource_gui, ImDisplay, PlateGrid)))
    
    
    def RUN_BATCH_ANALYZER(self, Meta_Data_df, displaygui, inout_resource_gui, ImDisplay, PlateGrid):
        
        ch1_spot_df, ch2_spot_df, ch3_spot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        ch4_spot_df, ch5_spot_df, cell_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        displaygui.setEnabled(False)
        columns = np.unique(np.asarray(Meta_Data_df['Column'], dtype=int))
        columns = columns[:1]
        rows = np.unique(np.asarray(Meta_Data_df['Row'], dtype=int))
        rows = rows[:2]
        fovs = np.unique(np.asarray(Meta_Data_df['FieldIndex'], dtype=int))
        fovs = fovs[:3]
        timepoints = np.unique(np.asarray(Meta_Data_df['TimePoint'], dtype=int))
        actionindices = np.unique(np.asarray(Meta_Data_df['ActionIndex'], dtype=int))
        actionindices = actionindices[:1]

        for col in columns:
            for row in rows:
                for fov in fovs:
                    for t in timepoints:
#                          for ai in actionindices:
                            ai =1
                        
                            df_checker = Meta_Data_df.loc[(Meta_Data_df['Column'] == str(col)) & 
                                                          (Meta_Data_df['Row'] == str(row)) & 
                                                          (Meta_Data_df['FieldIndex'] == str(fov)) & 
                                                          (Meta_Data_df['TimePoint'] == str(t))]
                                    
                            ImageForNucMask = self.IMG_FOR_NUC_MASK(df_checker)
                            
                            if ImageForNucMask.ndim ==2:
                            
                                nuc_bndry, nuc_mask = self.ImageAnalyzer.neuceli_segmenter(ImageForNucMask)
                                labeled_nuc, number_nuc = label(nuc_mask)
                                nuc_labels = np.unique(labeled_nuc)
                            
                                nuc_centroid_locations = ndimage.measurements.center_of_mass(nuc_mask, labeled_nuc, 
                                                                                             nuc_labels[nuc_labels>0])
                                data = { "Column": [col]*number_nuc, "Row": [row]*number_nuc, 
                                         "TimePoint": [t]*number_nuc, "FieldIndex": [fov]*number_nuc,
                                         "ZSlice": ["max project"]*number_nuc, "ActionIndex":[ai]*number_nuc}
                                df = pd.DataFrame(data)
                                regions = regionprops(labeled_nuc, ImageForNucMask)
                                props = regionprops_table(labeled_nuc, ImageForNucMask, properties=(
                                                        'centroid', 'orientation', 'major_axis_length', 'minor_axis_length',
                                                        'area', 'label' , 'max_intensity', 'min_intensity', 'mean_intensity',
                                                        'orientation', 'perimeter'))
                                pixpermicron = np.asarray(df_checker["PixPerMic"].iloc[0]).astype(float)
                                props_df = pd.DataFrame(props)
                                props_df['major_axis_length'] = pixpermicron*props_df['major_axis_length']
                                props_df['minor_axis_length'] = pixpermicron*props_df['minor_axis_length']
                                props_df['area'] = pixpermicron*pixpermicron*props_df['area']
                                props_df['perimeter'] = pixpermicron*props_df['perimeter']
                                image_cells_df = pd.concat([df,props_df], axis=1)
                                cell_df = pd.concat([cell_df, image_cells_df], axis=0, ignore_index=True)
                                
                                if self.AnalysisGui.NucMaskCheckBox.isChecked() == True:
                                    
                                    mask_file_name = ['Nuclei_Mask_for_Col' + str(col) + r'_row' + str(row)+
                                                      r'_Time' + str(t) + r'_Field' + str(fov) + r'.jpg']
                                    mask_full_name = os.path.join(inout_resource_gui.Output_dir, mask_file_name[0])
                                    cv2.imwrite(mask_full_name,nuc_mask)
                                
                            else:
                                
                                nuc_bndry, nuc_mask = self.Z_STACK_NUC_SEGMENTER(ImageForNucMask)
                                label_nuc_stack = self.Z_STACK_NUC_LABLER(ImageForLabel)

                            ch1_xyz, ch1_xyz_3D, ch2_xyz, ch2_xyz_3D, ch3_xyz, ch3_xyz_3D, ch4_xyz, ch4_xyz_3D, ch5_xyz, ch5_xyz_3D = self.IMAGE_FOR_SPOT_DETECTION( df_checker, ImageForNucMask)

                            if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:

                                if self.AnalysisGui.SpotMaxZProject.isChecked() == True:
                                            
                                        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:
                                            if ch1_xyz!=[]:
                                                ch1_xyz_round = np.round(np.asarray(ch1_xyz)).astype('int')
                                                ch1_spot_nuc_labels = labeled_nuc[ch1_xyz_round[:,0], ch1_xyz_round[:,1]]
                                                ch1_num_spots = ch1_xyz.__len__()

                                                data = { "Column": [col]*ch1_num_spots, "Row": [row]*ch1_num_spots, 
                                                         "TimePoint": [t]*ch1_num_spots, "FieldIndex": [fov]*ch1_num_spots,
                                                         "ZSlice": ["max project"]*ch1_num_spots, "Channel": [1]*ch1_num_spots,
                                                         "ActionIndex": [ai]*ch1_num_spots, "cell index": ch1_spot_nuc_labels,
                                                         "x_location": ch1_xyz[:,0], "y_location": ch1_xyz[:,1],
                                                         "z_location": ch1_xyz[:,2]}
                                                df_ch1 = pd.DataFrame(data)
                                                ch1_spot_df=pd.concat([ch1_spot_df,df_ch1],ignore_index=True)
                                                
                                        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
                                            if ch2_xyz!=[]:
                                                ch2_xyz_round = np.round(np.asarray(ch2_xyz)).astype('int')
                                                ch2_spot_nuc_labels = labeled_nuc[ch2_xyz_round[:,0], ch2_xyz_round[:,1]]
                                                ch2_num_spots = ch2_xyz.__len__()
                                                
                                                data = { "Column": [col]*ch2_num_spots, "Row": [row]*ch2_num_spots, 
                                                         "TimePoint": [t]*ch2_num_spots, "FieldIndex": [fov]*ch2_num_spots,
                                                         "ZSlice": ["max project"]*ch2_num_spots, "Channel": [2]*ch2_num_spots,
                                                         "ActionIndex": [ai]*ch2_num_spots, "cell index": ch2_spot_nuc_labels,
                                                         "x_location": ch2_xyz[:,0], "y_location": ch2_xyz[:,1],
                                                         "z_location": ch2_xyz[:,2]}
                                                df_ch2 = pd.DataFrame(data)
                                                ch2_spot_df=pd.concat([ch2_spot_df,df_ch2],ignore_index=True)
                                               
                                        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
                                            if ch3_xyz!=[]:

                                                ch3_xyz_round = np.round(np.asarray(ch3_xyz)).astype('int')
                                                ch3_spot_nuc_labels = labeled_nuc[ch3_xyz_round[:,0], ch3_xyz_round[:,1]]
                                                ch3_num_spots = ch3_xyz.__len__()
                                            
                                                data = { "Column": [col]*ch3_num_spots, "Row": [row]*ch3_num_spots, 
                                                         "TimePoint": [t]*ch3_num_spots, "FieldIndex": [fov]*ch3_num_spots,
                                                         "ZSlice": ["max project"]*ch3_num_spots, "Channel": [3]*ch3_num_spots,
                                                         "ActionIndex": [ai]*ch3_num_spots, "cell index": ch3_spot_nuc_labels,
                                                         "x_location": ch3_xyz[:,0], "y_location": ch3_xyz[:,1],
                                                         "z_location": ch3_xyz[:,2]}

                                                df_ch3 = pd.DataFrame(data)
                                                ch3_spot_df=pd.concat([ch3_spot_df, df_ch3],ignore_index=True)
                                                
                                        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
                                            if ch4_xyz!=[]:

                                                ch4_xyz_round = np.round(np.asarray(ch4_xyz)).astype('int')
                                                ch4_spot_nuc_labels = labeled_nuc[ch4_xyz_round[:,0], ch4_xyz_round[:,1]]
                                                ch4_num_spots = ch4_xyz.__len__()

                                                data = { "Column": [col]*ch4_num_spots, "Row": [row]*ch4_num_spots, 
                                                         "TimePoint": [t]*ch4_num_spots, "FieldIndex": [fov]*ch4_num_spots,
                                                         "ZSlice": ["max project"]*ch4_num_spots, "Channel": [4]*ch4_num_spots,
                                                         "ActionIndex": [ai]*ch4_num_spots, "cell index": ch4_spot_nuc_labels,
                                                         "x_location": ch4_xyz[:,0], "y_location": ch4_xyz[:,1],
                                                         "z_location": ch4_xyz[:,2]}

                                                df_ch4 = pd.DataFrame(data)
                                                ch4_spot_df=pd.concat([ch4_spot_df, df_ch4],ignore_index=True)
                                                

                                        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
                                            if ch5_xyz!=[]:

                                                ch5_xyz_round = np.round(np.asarray(ch5_xyz)).astype('int')
                                                ch5_spot_nuc_labels = labeled_nuc[ch5_xyz_round[:,0], ch5_xyz_round[:,1]]
                                                ch5_num_spots = ch5_xyz.__len__()
                                                

                                                data = { "Column": [col]*ch5_num_spots, "Row": [row]*ch5_num_spots, 
                                                         "TimePoint": [t]*ch5_num_spots, "FieldIndex": [fov]*ch5_num_spots,
                                                         "ZSlice": ["max project"]*ch5_num_spots, "Channel": [5]*ch5_num_spots,
                                                         "ActionIndex": [ai]*ch5_num_spots, "cell index": ch5_spot_nuc_labels,
                                                         "x_location": ch5_xyz[:,0], "y_location": ch5_xyz[:,1],
                                                         "z_location": ch5_xyz[:,2]}

                                                df_ch5 = pd.DataFrame(data)
                                                ch5_spot_df=pd.concat([ch5_spot_df, df_ch5],ignore_index=True)
                                                                                
                                                                                
        if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
            
            xlsx_name = ['Nuclei_Information.xlsx']
            xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
            
            cell_df.to_excel(xlsx_full_name)
        
        if ch1_spot_df.empty == False:
            
            if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch1_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
                ch1_spot_df.to_excel(xlsx_full_name)
            
        if ch2_spot_df.empty == False:
            
            if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch2_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
                ch2_spot_df.to_excel(xlsx_full_name)   
            
        if ch3_spot_df.empty == False:
            
            if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch3_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
                ch3_spot_df.to_excel(xlsx_full_name)   
            
        if ch4_spot_df.empty == False:
            
            if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch4_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
                ch4_spot_df.to_excel(xlsx_full_name) 
            
        if ch5_spot_df.empty == False:
            
            if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch5_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
                ch5_spot_df.to_excel(xlsx_full_name)   
            
        if self.AnalysisGui.SpotsDistance.isChecked() == True:
            
            spot_distances = self.Calculate_Spot_Distances( cell_df, ch1_spot_df, ch2_spot_df, ch3_spot_df, ch4_spot_df, ch5_spot_df, df_checker)
            
            coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
            xlsx_name = ['Spot_Distances_' + coordinates_method + r'.xlsx']
            xlsx_full_name = os.path.join(inout_resource_gui.Output_dir, xlsx_name[0])
            
            with pd.ExcelWriter(xlsx_full_name) as writer:  
                for key in spot_distances.keys():

                    spot_distances[key].to_excel(writer, sheet_name = key)
                            
                            
    def Calculate_Spot_Distances(self, cell_df, ch1_spot_df, ch2_spot_df, ch3_spot_df, ch4_spot_df, ch5_spot_df, df_checker):
        spot_distances = {}
        for index, df_row in cell_df.iterrows():
            
            
            col = np.asarray(df_row['Column'], dtype=int)
            row = np.asarray(df_row['Row'], dtype=int)
            timepoint = np.asarray(df_row['TimePoint'], dtype=int)
            fieldindex = np.asarray(df_row['FieldIndex'], dtype=int)
            zslice = df_row['ZSlice']
            actionindex = np.asarray(df_row['ActionIndex'], dtype=int)
            cell_label = np.asarray(df_row['label'], dtype=int)
            spot_pd_dict = {}
            
            if ch1_spot_df.empty == False:
                
                cell_spots_ch1 = ch1_spot_df.loc[(ch1_spot_df['Column'] == col) & (ch1_spot_df['Row'] == row) & 
                                      (ch1_spot_df['TimePoint'] == timepoint) & 
                                      (ch1_spot_df['FieldIndex'] == fieldindex)& 
                                      (ch1_spot_df['cell index']== cell_label )& 
                                      (ch1_spot_df['ActionIndex']== actionindex)&
                                      (ch1_spot_df['ZSlice']== zslice) ] 
                
                spot_pd_dict['ch1'] = cell_spots_ch1
                
                
            if ch2_spot_df.empty == False:
                
                cell_spots_ch2 = ch2_spot_df.loc[(ch2_spot_df['Column'] == col) & (ch2_spot_df['Row'] == row) & 
                                      (ch2_spot_df['TimePoint'] == timepoint) & 
                                      (ch2_spot_df['FieldIndex'] == fieldindex)& 
                                      (ch2_spot_df['cell index']== cell_label )& 
                                      (ch2_spot_df['ActionIndex']== actionindex)&
                                      (ch2_spot_df['ZSlice']== zslice) ] 
                
                spot_pd_dict['ch2'] = cell_spots_ch2

            if ch3_spot_df.empty == False:
                
                cell_spots_ch3 = ch3_spot_df.loc[(ch3_spot_df['Column'] == col) & (ch3_spot_df['Row'] == row) & 
                                      (ch3_spot_df['TimePoint'] == timepoint) & 
                                      (ch3_spot_df['FieldIndex'] == fieldindex)& 
                                      (ch3_spot_df['cell index'] == cell_label )& 
                                      (ch3_spot_df['ActionIndex'] == actionindex)&
                                      (ch3_spot_df['ZSlice'] == zslice) ] 
                
                spot_pd_dict['ch3'] = cell_spots_ch3
                        
            if ch4_spot_df.empty == False:
                
                cell_spots_ch4 = ch4_spot_df.loc[(ch4_spot_df['Column'] == col) & (ch4_spot_df['Row'] == row) & 
                                      (ch4_spot_df['TimePoint'] == timepoint) & 
                                      (ch4_spot_df['FieldIndex'] == fieldindex)& 
                                      (ch4_spot_df['cell index'] == cell_label )& 
                                      (ch4_spot_df['ActionIndex'] == actionindex)&
                                      (ch4_spot_df['ZSlice'] == zslice)] 
                
                spot_pd_dict['ch4'] = cell_spots_ch4
                
            if ch5_spot_df.empty == False:
                
                cell_spots_ch5 = ch5_spot_df.loc[(ch5_spot_df['Column'] == col) & (ch5_spot_df['Row'] == row) & 
                                      (ch5_spot_df['TimePoint'] == timepoint) & 
                                      (ch5_spot_df['FieldIndex'] == fieldindex)& 
                                      (ch5_spot_df['cell index'] == cell_label)& 
                                      (ch5_spot_df['ActionIndex'] == actionindex)&
                                      (ch5_spot_df['ZSlice'] == zslice)] 
                
                spot_pd_dict['ch5'] = cell_spots_ch5
            
            ch_distances = []
                   
            for key1 in spot_pd_dict.keys():
                for key2 in spot_pd_dict.keys():
                    
                    ch_distance1 = key2 + r'_' + key1
                    if ch_distance1 in ch_distances:
                        pass
                    else:
                        ch_distance = key1 + r'_' + key2
                        ch_distances.append(ch_distance)
                        
                        if ch_distance not in spot_distances:
                            spot_distances[ch_distance] = pd.DataFrame()
                            
                        dist_pd = self.DISTANCE_calculator(spot_pd_dict,key1,key2, df_checker)

                        num_distances = dist_pd.__len__()          
                        data = { "Column": [col]*num_distances, "Row": [row]*num_distances, 
                                 "TimePoint": [timepoint]*num_distances, "FieldIndex": [fieldindex]*num_distances,
                                 "ZSlice": ["max project"]*num_distances, "ActionIndex": [actionindex]*num_distances, 
                                 "cell index": [cell_label]*num_distances}

                        temp_df = pd.DataFrame(data)
                        
                        temp_df = pd.concat([temp_df, dist_pd],axis=1)
                        spot_distances[ch_distance]=pd.concat([spot_distances[ch_distance], temp_df],ignore_index=True)

        return spot_distances            
                    
                                                        
    def DISTANCE_calculator(self, spot_pd_dict, key1, key2, df_checker):
        
        loci1 = spot_pd_dict[key1]
        loci2 = spot_pd_dict[key2]
        dist_pd = pd.DataFrame()
        
        for ind1, locus1 in loci1.iterrows():
            for ind2, locus2 in loci2.iterrows():

                dist_2d =  math.sqrt(
                                    math.pow((locus1['x_location'] - locus2['x_location']),2) +
                                    math.pow((locus1['y_location'] - locus2['y_location']),2) 
                                    )

                dist_3d =  math.sqrt(
                                    math.pow((locus1['x_location'] - locus2['x_location']),2) +
                                    math.pow((locus1['y_location'] - locus2['y_location']),2) +
                                    math.pow((locus1['z_location'] - locus2['z_location']),2)
                                    )
                
                s1 = key1 + '_spot_index(1)'
                s2 = key2 + '_spot_index(2)'
                data = { str(s1): [ind1], str(s2): [ind2], 
                        'XY-Disance(pixels)': [dist_2d], 'XYZ-Distnce(pixels)': [dist_3d],
                        'XY-Disance(micron)': [dist_2d*np.asarray(df_checker["PixPerMic"].iloc[0]).astype(float)], 
                        'XYZ-Distnce(micron)':[dist_3d*np.asarray(df_checker["PixPerMic"].iloc[0]).astype(float)]}
                temp_df = pd.DataFrame(data)
                dist_pd = pd.concat([dist_pd, temp_df], ignore_index=True)

        return dist_pd
                
                
                

    def IMG_FOR_NUC_MASK(self, df_checker):
        
        if df_checker.empty == False:

            if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:

                maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
                imgformask = df_checker.loc[(df_checker['Channel'] == maskchannel)]
                loadedimg_formask = self.ImageAnalyzer.max_z_project(imgformask)
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
            nuc_bndry_single, nuc_mask_single = self.ImageAnalyzer.neuceli_segmenter(single_z_img)

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
            coordinates = self.ImageAnalyzer.SpotDetector(im_uint8, self.AnalysisGui, ImageForNucMask)
            
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
        coordinates_max_project = self.ImageAnalyzer.SpotDetector(im_uint8, self.AnalysisGui, ImageForNucMask)
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
    
    
    
    
    
    
    
    