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
    
    def __init__(self,analysisgui, image_analyzer, inout_resource_gui):

        self.inout_resource_gui = inout_resource_gui
        self.AnalysisGui = analysisgui
        self.ImageAnalyzer = image_analyzer
        self.ch1_spot_df, self.ch2_spot_df, self.ch3_spot_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.ch4_spot_df, self.ch5_spot_df, self.cell_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_checker = pd.DataFrame()
        self.Meta_Data_df = pd.DataFrame()
        self.spot_distances = {}
        self.experiment_name = []
        self.output_prefix = []
        self.output_folder = []
    
    def ON_APPLYBUTTON(self, Meta_Data_df):
        self.Meta_Data_df = Meta_Data_df
        path_list = os.path.split(self.Meta_Data_df["ImageName"][0])[0].split(r'/')
        self.experiment_name = path_list[path_list.__len__()-2]
        self.output_prefix  = path_list[path_list.__len__()-1]
        self.output_folder = os.path.join(self.inout_resource_gui.Output_dir,self.experiment_name)
        if os.path.isdir(self.output_folder) == False:
            os.mkdir(self.output_folder)            
            
        columns = np.unique(np.asarray(self.Meta_Data_df['column'], dtype=int))
        columns = columns [:1]
        rows = np.unique(np.asarray(self.Meta_Data_df['row'], dtype=int))
        rows = rows[:1]
        fovs = np.unique(np.asarray(self.Meta_Data_df['field_index'], dtype=int))
        fovs = fovs[:2]
        time_points = np.unique(np.asarray(self.Meta_Data_df['time_point'], dtype=int))
        actionindices = np.unique(np.asarray(self.Meta_Data_df['action_index'], dtype=int))
        
        jobs_number=self.inout_resource_gui.NumCPUsSpinBox.value()
        Parallel(n_jobs=jobs_number, prefer="threads")(delayed(self.BATCH_ANALYZER)(col,row,fov,t) for t in time_points for fov in fovs for row in rows for col in columns)
        
        xlsx_output_folder = os.path.join(self.output_folder, 'whole_plate_resutls')
        if os.path.isdir(xlsx_output_folder) == False:
            os.mkdir(xlsx_output_folder) 
                                                                                                
        if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
            
            xlsx_name = ['Nuclei_Information.xlsx']
            xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
            
            self.cell_df.to_excel(xlsx_full_name)
            
            well_nuc_folder = os.path.join(self.output_folder, 'well_nuclei_resutls')
            if os.path.isdir(well_nuc_folder) == False:
                os.mkdir(well_nuc_folder)
            for col in columns:
                for row in rows:
                    well_nuc_df = self.cell_df.loc[(self.cell_df['column'] == col) & (self.cell_df['row'] == row)]
                    well_nuc_filename = self.output_prefix + '_nuclei_information_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    nuc_well_csv_full_name = os.path.join(well_nuc_folder, well_nuc_filename)
                    well_nuc_df.to_csv(path_or_buf=nuc_well_csv_full_name, encoding='utf8')

        if self.ch1_spot_df.empty == False:
            
            if self.AnalysisGui.SpotsLocation.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch1_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
                self.ch1_spot_df.to_excel(xlsx_full_name)
                
                well_spot_loc_folder = os.path.join(self.output_folder, 'well_spots_locations')
                if os.path.isdir(well_spot_loc_folder) == False:
                    os.mkdir(well_spot_loc_folder)
                for col in columns:
                    for row in rows:
                        spot_loc_df = self.ch1_spot_df.loc[(self.ch1_spot_df['column'] == col) & (self.ch1_spot_df['row'] == row)]
                        spot_loc_filename = self.output_prefix + '_ch1_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                        spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                        spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')
                
                
            
        if self.ch2_spot_df.empty == False:
            
            if self.AnalysisGui.SpotsLocation.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch2_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
                self.ch2_spot_df.to_excel(xlsx_full_name)   
                
                well_spot_loc_folder = os.path.join(self.output_folder, 'well_spots_locations')
                if os.path.isdir(well_spot_loc_folder) == False:
                    os.mkdir(well_spot_loc_folder)
                for col in columns:
                    for row in rows:
                        spot_loc_df = self.ch2_spot_df.loc[(self.ch2_spot_df['column'] == col) & (self.ch2_spot_df['row'] == row)]
                        spot_loc_filename = self.output_prefix + '_ch2_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                        spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                        spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')
            
        if self.ch3_spot_df.empty == False:
            
            if self.AnalysisGui.SpotsLocation.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch3_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
                self.ch3_spot_df.to_excel(xlsx_full_name)   
                
                well_spot_loc_folder = os.path.join(self.output_folder, 'well_spots_locations')
                if os.path.isdir(well_spot_loc_folder) == False:
                    os.mkdir(well_spot_loc_folder)
                for col in columns:
                    for row in rows:
                        spot_loc_df = self.ch3_spot_df.loc[(self.ch3_spot_df['column'] == col) & (self.ch3_spot_df['row'] == row)]
                        spot_loc_filename = self.output_prefix + '_ch3_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                        spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                        spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')
            
        if self.ch4_spot_df.empty == False:
            
            if self.AnalysisGui.SpotsLocation.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch4_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
                self.ch4_spot_df.to_excel(xlsx_full_name) 
                
                well_spot_loc_folder = os.path.join(self.output_folder, 'well_spots_locations')
                if os.path.isdir(well_spot_loc_folder) == False:
                    os.mkdir(well_spot_loc_folder)
                for col in columns:
                    for row in rows:
                        spot_loc_df = self.ch4_spot_df.loc[(self.ch4_spot_df['column'] == col) & (self.ch4_spot_df['row'] == row)]
                        spot_loc_filename = self.output_prefix + '_ch4_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                        spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                        spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')
            
        if self.ch5_spot_df.empty == False:
            
            if self.AnalysisGui.SpotsLocation.isChecked() == True:
                coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
                xlsx_name = ['Ch5_Spot_Locations_' + coordinates_method + r'.xlsx']
                xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
                self.ch5_spot_df.to_excel(xlsx_full_name)   
                
                well_spot_loc_folder = os.path.join(self.output_folder, 'well_spots_locations')
                if os.path.isdir(well_spot_loc_folder) == False:
                    os.mkdir(well_spot_loc_folder)
                for col in columns:
                    for row in rows:
                        spot_loc_df = self.ch5_spot_df.loc[(self.ch5_spot_df['column'] == col) & (self.ch5_spot_df['row'] == row)]
                        spot_loc_filename = self.output_prefix + '_ch5_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                        spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                        spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')
            
        if self.AnalysisGui.SpotsDistance.isChecked() == True:
            
            
            Parallel(n_jobs=jobs_number, prefer="threads")(delayed(self.Calculate_Spot_Distances)(index, df_row) for index, df_row in self.cell_df.iterrows())
            coordinates_method = self.AnalysisGui.SpotLocationCbox.currentText()
            xlsx_name = ['Spot_Distances_' + coordinates_method + r'.xlsx']
            xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
            
            with pd.ExcelWriter(xlsx_full_name) as writer:  
                for key in self.spot_distances.keys():
#                     self.spot_distances[key].sort_values(by=['column', 'row', 'field_index', 
#                                                              'time_point', 'action_index', 'cell index'])

                    self.spot_distances[key].to_excel(writer, sheet_name = key)
            
        well_spot_dist_folder = os.path.join(self.output_folder, 'well_spots_distances')
        if os.path.isdir(well_spot_dist_folder) == False:
            os.mkdir(well_spot_dist_folder)
        for key in self.spot_distances.keys():
            for col in columns:
                for row in rows:
                    spot_dist_df = self.spot_distances[key].loc[(self.spot_distances[key]['column'] == col) & 
                                                               (self.spot_distances[key]['row'] == row)]
                    spot_dist_filename = self.output_prefix + '_' + key +'_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    spot_dist_well_csv_full_name = os.path.join(well_spot_dist_folder, spot_dist_filename)
                    spot_dist_df.to_csv(path_or_buf=spot_dist_well_csv_full_name, encoding='utf8')
                    
    def BATCH_ANALYZER(self, col,row,fov,t): 
        ai = 1
        
        self.df_checker = self.Meta_Data_df.loc[(self.Meta_Data_df['column'] == str(col)) & 
                                      (self.Meta_Data_df['row'] == str(row)) & 
                                      (self.Meta_Data_df['field_index'] == str(fov)) & 
                                      (self.Meta_Data_df['time_point'] == str(t))]

        ImageForNucMask = self.IMG_FOR_NUC_MASK()

        if ImageForNucMask.ndim ==2:

            nuc_bndry, nuc_mask = self.ImageAnalyzer.neuceli_segmenter(ImageForNucMask)
            labeled_nuc, number_nuc = label(nuc_mask)
            nuc_labels = np.unique(labeled_nuc)

            nuc_centroid_locations = ndimage.measurements.center_of_mass(nuc_mask, labeled_nuc, 
                                                                         nuc_labels[nuc_labels>0])
            data = { "column": [col]*number_nuc, "row": [row]*number_nuc, 
                     "time_point": [t]*number_nuc, "field_index": [fov]*number_nuc,
                     "z_slice": ["max_project"]*number_nuc, "action_index":[ai]*number_nuc}
            df = pd.DataFrame(data)
            regions = regionprops(labeled_nuc, ImageForNucMask)
            props = regionprops_table(labeled_nuc, ImageForNucMask, properties=(
                                    'label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length',
                                    'area', 'max_intensity', 'min_intensity', 'mean_intensity',
                                    'orientation', 'perimeter'))
            pixpermicron = np.asarray(self.df_checker["PixPerMic"].iloc[0]).astype(float)
            props_df = pd.DataFrame(props)
            props_df.rename(columns={ "label":"cell_index", "centroid-0":"centroid_x", "centroid-1":"centroid_y"})
            props_df['major_axis_length'] = pixpermicron*props_df['major_axis_length']
            props_df['minor_axis_length'] = pixpermicron*props_df['minor_axis_length']
            props_df['area'] = pixpermicron*pixpermicron*props_df['area']
            props_df['perimeter'] = pixpermicron*props_df['perimeter']
            image_cells_df = pd.concat([df,props_df], axis=1)
            self.cell_df = pd.concat([self.cell_df, image_cells_df], axis=0, ignore_index=True)
            
            
            
            if self.AnalysisGui.NucMaskCheckBox.isChecked() == True:
                nuc_mask_output_folder = os.path.join(self.output_folder, 'nuclei_masks')
                if os.path.isdir(nuc_mask_output_folder) == False:
                    os.mkdir(nuc_mask_output_folder) 

                mask_file_name = ['Nuclei_Mask_for_Col' + str(col) + r'_row' + str(row)+
                                  r'_Time' + str(t) + r'_Field' + str(fov) + r'.jpg']
                mask_full_name = os.path.join(nuc_mask_output_folder, mask_file_name[0])
                cv2.imwrite(mask_full_name,nuc_mask)

        else:

            nuc_bndry, nuc_mask = self.Z_STACK_NUC_SEGMENTER(ImageForNucMask)
            label_nuc_stack = self.Z_STACK_NUC_LABLER(ImageForLabel)

        ch1_xyz, ch1_xyz_3D, ch2_xyz, ch2_xyz_3D, ch3_xyz, ch3_xyz_3D, ch4_xyz, ch4_xyz_3D, ch5_xyz, ch5_xyz_3D = self.IMAGE_FOR_SPOT_DETECTION( ImageForNucMask)

        if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:

            if self.AnalysisGui.SpotMaxZProject.isChecked() == True:

                    if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:
                        if ch1_xyz!=[]:
                            ch1_xyz_round = np.round(np.asarray(ch1_xyz)).astype('int')
                            ch1_spot_nuc_labels = labeled_nuc[ch1_xyz_round[:,0], ch1_xyz_round[:,1]]
                            ch1_num_spots = ch1_xyz.__len__()

                            data = { "column": [col]*ch1_num_spots, "row": [row]*ch1_num_spots, 
                                     "time_point": [t]*ch1_num_spots, "field_index": [fov]*ch1_num_spots,
                                     "z_slice": ["max_project"]*ch1_num_spots, "channel": [1]*ch1_num_spots,
                                     "action_index": [ai]*ch1_num_spots, "cell_index": ch1_spot_nuc_labels,
                                     "x_location": ch1_xyz[:,0], "y_location": ch1_xyz[:,1],
                                     "z_location": ch1_xyz[:,2]}
                            df_ch1 = pd.DataFrame(data)
                            
                            cell_indices = np.unique(ch1_spot_nuc_labels)
                            if self.AnalysisGui.SpotPerCh1SpinBox.value()>0:
                                for ind in cell_indices:
                                    temp_pd_df = df_ch1.loc[df_ch1["cell_index"] == ind]
                                    if temp_pd_df.__len__()!= self.AnalysisGui.SpotPerCh1SpinBox.value():
                            
                                        df_ch1 = df_ch1[df_ch1.cell_index != ind]
                                
                            self.ch1_spot_df=pd.concat([self.ch1_spot_df,df_ch1],ignore_index=True)

                    if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
                        if ch2_xyz!=[]:
                            ch2_xyz_round = np.round(np.asarray(ch2_xyz)).astype('int')
                            ch2_spot_nuc_labels = labeled_nuc[ch2_xyz_round[:,0], ch2_xyz_round[:,1]]
                            ch2_num_spots = ch2_xyz.__len__()

                            data = { "column": [col]*ch2_num_spots, "row": [row]*ch2_num_spots, 
                                     "time_point": [t]*ch2_num_spots, "field_index": [fov]*ch2_num_spots,
                                     "z_slice": ["max_project"]*ch2_num_spots, "channel": [2]*ch2_num_spots,
                                     "action_index": [ai]*ch2_num_spots, "cell_index": ch2_spot_nuc_labels,
                                     "x_location": ch2_xyz[:,0], "y_location": ch2_xyz[:,1],
                                     "z_location": ch2_xyz[:,2]}
                            df_ch2 = pd.DataFrame(data)
                            
                            cell_indices = np.unique(ch2_spot_nuc_labels)
                            if self.AnalysisGui.SpotPerCh2SpinBox.value()>0:
                                for ind in cell_indices:
                                    temp_pd_df = df_ch2.loc[df_ch2["cell_index"] == ind]
                                    if temp_pd_df.__len__()!= self.AnalysisGui.SpotPerCh2SpinBox.value():
                            
                                        df_ch2 = df_ch2[df_ch2.cell_index != ind]
                            
                            self.ch2_spot_df=pd.concat([self.ch2_spot_df,df_ch2],ignore_index=True)

                    if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
                        if ch3_xyz!=[]:

                            ch3_xyz_round = np.round(np.asarray(ch3_xyz)).astype('int')
                            ch3_spot_nuc_labels = labeled_nuc[ch3_xyz_round[:,0], ch3_xyz_round[:,1]]
                            ch3_num_spots = ch3_xyz.__len__()

                            data = { "column": [col]*ch3_num_spots, "row": [row]*ch3_num_spots, 
                                     "time_point": [t]*ch3_num_spots, "field_index": [fov]*ch3_num_spots,
                                     "z_slice": ["max_project"]*ch3_num_spots, "channel": [3]*ch3_num_spots,
                                     "action_index": [ai]*ch3_num_spots, "cell_index": ch3_spot_nuc_labels,
                                     "x_location": ch3_xyz[:,0], "y_location": ch3_xyz[:,1],
                                     "z_location": ch3_xyz[:,2]}

                            df_ch3 = pd.DataFrame(data)
                            
                            cell_indices = np.unique(ch3_spot_nuc_labels)
                            if self.AnalysisGui.SpotPerCh3SpinBox.value()>0:
                                for ind in cell_indices:
                                    temp_pd_df = df_ch3.loc[df_ch3["cell_index"] == ind]
                                    if temp_pd_df.__len__()!= self.AnalysisGui.SpotPerCh3SpinBox.value():
                            
                                        df_ch3 = df_ch3[df_ch3.cell_index != ind]
                                
                            self.ch3_spot_df=pd.concat([self.ch3_spot_df, df_ch3],ignore_index=True)

                    if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
                        if ch4_xyz!=[]:

                            ch4_xyz_round = np.round(np.asarray(ch4_xyz)).astype('int')
                            ch4_spot_nuc_labels = labeled_nuc[ch4_xyz_round[:,0], ch4_xyz_round[:,1]]
                            ch4_num_spots = ch4_xyz.__len__()

                            data = { "column": [col]*ch4_num_spots, "row": [row]*ch4_num_spots, 
                                     "time_point": [t]*ch4_num_spots, "field_index": [fov]*ch4_num_spots,
                                     "z_slice": ["max_project"]*ch4_num_spots, "channel": [4]*ch4_num_spots,
                                     "action_index": [ai]*ch4_num_spots, "cell_index": ch4_spot_nuc_labels,
                                     "x_location": ch4_xyz[:,0], "y_location": ch4_xyz[:,1],
                                     "z_location": ch4_xyz[:,2]}

                            df_ch4 = pd.DataFrame(data)
                            
                            cell_indices = np.unique(ch4_spot_nuc_labels)
                            if self.AnalysisGui.SpotPerCh4SpinBox.value()>0:
                                for ind in cell_indices:
                                    temp_pd_df = df_ch4.loc[df_ch4["cell_index"] == ind]
                                    if temp_pd_df.__len__()!= self.AnalysisGui.SpotPerCh4SpinBox.value():
                            
                                        df_ch4 = df_ch4[df_ch4.cell_index != ind]
                                
                            self.ch4_spot_df=pd.concat([self.ch4_spot_df, df_ch4],ignore_index=True)


                    if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
                        if ch5_xyz!=[]:

                            ch5_xyz_round = np.round(np.asarray(ch5_xyz)).astype('int')
                            ch5_spot_nuc_labels = labeled_nuc[ch5_xyz_round[:,0], ch5_xyz_round[:,1]]
                            ch5_num_spots = ch5_xyz.__len__()


                            data = { "column": [col]*ch5_num_spots, "row": [row]*ch5_num_spots, 
                                     "time_point": [t]*ch5_num_spots, "field_index": [fov]*ch5_num_spots,
                                     "z_slice": ["max_project"]*ch5_num_spots, "channel": [5]*ch5_num_spots,
                                     "action_index": [ai]*ch5_num_spots, "cell_index": ch5_spot_nuc_labels,
                                     "x_location": ch5_xyz[:,0], "y_location": ch5_xyz[:,1],
                                     "z_location": ch5_xyz[:,2]}

                            df_ch5 = pd.DataFrame(data)
                            
                            cell_indices = np.unique(ch5_spot_nuc_labels)
                            if self.AnalysisGui.SpotPerCh5SpinBox.value()>0:
                                for ind in cell_indices:
                                    temp_pd_df = df_ch5.loc[df_ch5["cell_index"] == ind]
                                    if temp_pd_df.__len__()!= self.AnalysisGui.SpotPerCh5SpinBox.value():
                            
                                        df_ch5 = df_ch5[df_ch5.cell_index != ind]
                                
                            self.ch5_spot_df=pd.concat([self.ch5_spot_df, df_ch5],ignore_index=True) 
                            
    def Calculate_Spot_Distances(self, index, df_row):
           
        col = np.asarray(df_row['column'], dtype=int)
        row = np.asarray(df_row['row'], dtype=int)
        time_point = np.asarray(df_row['time_point'], dtype=int)
        field_index = np.asarray(df_row['field_index'], dtype=int)
        z_slice = df_row['z_slice']
        action_index = np.asarray(df_row['action_index'], dtype=int)
        cell_label = np.asarray(df_row['label'], dtype=int)
        spot_pd_dict = {}

        if self.ch1_spot_df.empty == False:

            cell_spots_ch1 = self.ch1_spot_df.loc[(self.ch1_spot_df['column'] == col) & (self.ch1_spot_df['row'] == row) & 
                                  (self.ch1_spot_df['time_point'] == time_point) & 
                                  (self.ch1_spot_df['field_index'] == field_index)& 
                                  (self.ch1_spot_df['cell_index']== cell_label )& 
                                  (self.ch1_spot_df['action_index']== action_index)&
                                  (self.ch1_spot_df['z_slice']== z_slice) ] 

            spot_pd_dict['ch1'] = cell_spots_ch1


        if self.ch2_spot_df.empty == False:

            cell_spots_ch2 = self.ch2_spot_df.loc[(self.ch2_spot_df['column'] == col) & (self.ch2_spot_df['row'] == row) & 
                                  (self.ch2_spot_df['time_point'] == time_point) & 
                                  (self.ch2_spot_df['field_index'] == field_index)& 
                                  (self.ch2_spot_df['cell_index']== cell_label )& 
                                  (self.ch2_spot_df['action_index']== action_index)&
                                  (self.ch2_spot_df['z_slice']== z_slice) ] 

            spot_pd_dict['ch2'] = cell_spots_ch2

        if self.ch3_spot_df.empty == False:

            cell_spots_ch3 = self.ch3_spot_df.loc[(self.ch3_spot_df['column'] == col) & (self.ch3_spot_df['row'] == row) & 
                                  (self.ch3_spot_df['time_point'] == time_point) & 
                                  (self.ch3_spot_df['field_index'] == field_index)& 
                                  (self.ch3_spot_df['cell_index'] == cell_label )& 
                                  (self.ch3_spot_df['action_index'] == action_index)&
                                  (self.ch3_spot_df['z_slice'] == z_slice) ] 

            spot_pd_dict['ch3'] = cell_spots_ch3

        if self.ch4_spot_df.empty == False:

            cell_spots_ch4 = self.ch4_spot_df.loc[(self.ch4_spot_df['column'] == col) & (self.ch4_spot_df['row'] == row) & 
                                  (self.ch4_spot_df['time_point'] == time_point) & 
                                  (self.ch4_spot_df['field_index'] == field_index)& 
                                  (self.ch4_spot_df['cell_index'] == cell_label )& 
                                  (self.ch4_spot_df['action_index'] == action_index)&
                                  (self.ch4_spot_df['z_slice'] == z_slice)] 

            spot_pd_dict['ch4'] = cell_spots_ch4

        if self.ch5_spot_df.empty == False:

            cell_spots_ch5 = self.ch5_spot_df.loc[(self.ch5_spot_df['column'] == col) & (self.ch5_spot_df['row'] == row) & 
                                  (self.ch5_spot_df['time_point'] == time_point) & 
                                  (self.ch5_spot_df['field_index'] == field_index)& 
                                  (self.ch5_spot_df['cell_index'] == cell_label)& 
                                  (self.ch5_spot_df['action_index'] == action_index)&
                                  (self.ch5_spot_df['z_slice'] == z_slice)] 

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

                    if ch_distance not in self.spot_distances:
                        self.spot_distances[ch_distance] = pd.DataFrame()

                    dist_pd = self.DISTANCE_calculator(spot_pd_dict,key1,key2)

                    num_distances = dist_pd.__len__()          
                    data = { "column": [col]*num_distances, "row": [row]*num_distances, 
                             "time_point": [time_point]*num_distances, "field_index": [field_index]*num_distances,
                             "z_slice": ["max_project"]*num_distances, "action_index": [action_index]*num_distances, 
                             "cell_index": [cell_label]*num_distances}

                    temp_df = pd.DataFrame(data)

                    temp_df = pd.concat([temp_df, dist_pd],axis=1)
                    self.spot_distances[ch_distance]=pd.concat([self.spot_distances[ch_distance], temp_df],ignore_index=True)

                 
                    
                                                        
    def DISTANCE_calculator(self, spot_pd_dict, key1, key2):
        
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
                        'XY-Disance(micron)': [dist_2d*np.asarray(self.df_checker["PixPerMic"].iloc[0]).astype(float)], 
                        'XYZ-Distnce(micron)':[dist_3d*np.asarray(self.df_checker["PixPerMic"].iloc[0]).astype(float)]}
                temp_df = pd.DataFrame(data)
                dist_pd = pd.concat([dist_pd, temp_df], ignore_index=True)

        return dist_pd
                
                
                

    def IMG_FOR_NUC_MASK(self):
        
        if self.df_checker.empty == False:

            if self.AnalysisGui.NucMaxZprojectCheckBox.isChecked() == True:

                maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]
                loadedimg_formask = self.ImageAnalyzer.max_z_project(imgformask)
                ImageForNucMask = cv2.normalize(loadedimg_formask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            else:
                z_imglist=[]
                maskchannel = str(self.AnalysisGui.NucleiChannel.currentIndex()+1)
                imgformask = self.df_checker.loc[(self.df_checker['channel'] == maskchannel)]

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
            nuc_bndry_single, nuc_mask_single = self.ImageAnalyzer.neuceli_segmenter(single_z_img,
                                                                                     self.Meta_Data_df["PixPerMic"].iloc[0])

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
    
    
    def IMAGE_FOR_SPOT_DETECTION(self, ImageForNucMask):
        
        ch1_xyz, ch2_xyz, ch3_xyz, ch4_xyz, ch5_xyz = [],[],[],[],[]
        ch1_xyz_3D, ch2_xyz_3D, ch3_xyz_3D, ch4_xyz_3D, ch5_xyz_3D = [],[],[],[],[]
        
        if self.AnalysisGui.SpotCh1CheckBox.isChecked() == True:

            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '1')]
            ch1_xyz, ch1_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
                
        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '2')]
            ch2_xyz, ch2_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
                
        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '3')]
            ch3_xyz, ch3_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
                    
        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '4')]
            ch4_xyz, ch4_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
        
        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '5')]
            ch5_xyz, ch5_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask)
            
        return ch1_xyz, ch1_xyz_3D, ch2_xyz, ch2_xyz_3D, ch3_xyz, ch3_xyz_3D, ch4_xyz, ch4_xyz_3D, ch5_xyz, ch5_xyz_3D

    
    def XYZ_SPOT_COORDINATES(self, images_pd_df, ImageForNucMask):
        
        z_imglist = []
        coordinates_list = []
        print(images_pd_df)
        for index, row in images_pd_df.iterrows():
            im = mpimg.imread(row['ImageName'])
            _z_coordinates1 = np.asarray(row['z_slice']).astype('float')
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

                spots_z_coordinates[i] = np.asarray(images_pd_df.loc[images_pd_df['z_slice']== str(spots_z_slices[i]+1)]
                                                 ['z_coordinate'].iloc[0]).astype('float')

            xyz_coordinates = np.append(np.asarray(coordinates_max_project).astype('float'), spots_z_coordinates, 1)
        else:
            
            xyz_coordinates = []
            
            
        return xyz_coordinates, coordinates_stack
    
    
    
    
    
    
    
    