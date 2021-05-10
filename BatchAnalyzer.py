import numpy as np
import cv2
import math
import os
import time
from skimage.measure import regionprops, regionprops_table
from scipy import ndimage
from PIL import Image
import pandas as pd
import matplotlib.image as mpimg
from PIL import Image, ImageQt
from scipy.ndimage import label, distance_transform_edt
import multiprocessing
from joblib import Parallel, delayed
import dill as pickle
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class BatchAnalysis(object):
    output_folder = []
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
    
    def ON_APPLYBUTTON(self, Meta_Data_df):
        seconds1 = time.time()
        while self.inout_resource_gui.Output_dir ==[]:
                
            self.inout_resource_gui.OUTPUT_FOLDER_LOADBTN()
            
        self.Meta_Data_df = Meta_Data_df
        path_list = os.path.split(self.Meta_Data_df["ImageName"][0])[0].split(r'/')
        self.experiment_name = path_list[path_list.__len__()-2]
        self.output_prefix  = path_list[path_list.__len__()-1]
        self.output_folder = os.path.join(self.inout_resource_gui.Output_dir,self.experiment_name)
        if os.path.isdir(self.output_folder) == False:
            os.mkdir(self.output_folder) 
            
        csv_config_folder = os.path.join(self.output_folder, 'configuration_files')
        if os.path.isdir(csv_config_folder) == False:
            os.mkdir(csv_config_folder) 
        self.config_file = os.path.join(csv_config_folder, 'analysis_configuration.csv')
        self.AnalysisGui.SAVE_CONFIGURATION(self.config_file, self.ImageAnalyzer)
        
        columns = np.unique(np.asarray(self.Meta_Data_df['column'], dtype=int))
        rows = np.unique(np.asarray(self.Meta_Data_df['row'], dtype=int))
        fovs = np.unique(np.asarray(self.Meta_Data_df['field_index'], dtype=int))
        time_points = np.unique(np.asarray(self.Meta_Data_df['time_point'], dtype=int))
        actionindices = np.unique(np.asarray(self.Meta_Data_df['action_index'], dtype=int))
        
        jobs_number=self.inout_resource_gui.NumCPUsSpinBox.value()
        Parallel(n_jobs=jobs_number, prefer="threads")(delayed(self.BATCH_ANALYZER)(col,row,fov,t) for t in time_points for fov in fovs for row in rows for col in columns)
#         with Pool(jobs_number) as p:
#             p.map(self.BATCH_ANALYZER, columns, rows, fovs, time_points)
            
        xlsx_output_folder = os.path.join(self.output_folder, 'whole_plate_resutls')
        if os.path.isdir(xlsx_output_folder) == False:
            os.mkdir(xlsx_output_folder) 
                                                                                                
        if self.AnalysisGui.NucInfoChkBox.isChecked() == True:
            
            xlsx_name = ['Nuclei_Information.xlsx']
            xlsx_full_name = os.path.join(xlsx_output_folder, xlsx_name[0])
            self.cell_df.rename(columns={ "label":"cell_index"})
            self.cell_df.to_excel(xlsx_full_name)
            
            well_nuc_folder = os.path.join(self.output_folder, 'well_nuclei_results')
            if os.path.isdir(well_nuc_folder) == False:
                os.mkdir(well_nuc_folder)
            for col in columns:
                for row in rows:
                    well_nuc_df = self.cell_df.loc[(self.cell_df['column'] == col) & (self.cell_df['row'] == row)]
                    if well_nuc_df.empty == False:
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
                        if spot_loc_df.empty == False:
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
                        if spot_loc_df.empty == False:
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
                        if spot_loc_df.empty == False:
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
                        if spot_loc_df.empty == False:
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
                        if spot_loc_df.empty == False:
                            spot_loc_filename = self.output_prefix + '_ch5_spots_locations_well_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                            spot_loc_well_csv_full_name = os.path.join(well_spot_loc_folder, spot_loc_filename)
                            spot_loc_df.to_csv(path_or_buf=spot_loc_well_csv_full_name, encoding='utf8')
            
        if self.AnalysisGui.SpotsDistance.isChecked() == True:
            columns = np.unique(np.asarray(self.cell_df['column'], dtype=int))
            rows = np.unique(np.asarray(self.cell_df['row'], dtype=int))
            Parallel(n_jobs=jobs_number, prefer="threads")(delayed(self.Calculate_Spot_Distances)(row, col) for row in rows for col in columns)

        seconds2 = time.time()
        
        diff=seconds2-seconds1
        print('Total Processing Time (Minutes):',diff/60)
        
    def BATCH_ANALYZER(self, col,row,fov,t): 
        ai = 1
        
        self.df_checker = self.Meta_Data_df.loc[(self.Meta_Data_df['column'] == str(col)) & 
                                      (self.Meta_Data_df['row'] == str(row)) & 
                                      (self.Meta_Data_df['field_index'] == str(fov)) & 
                                      (self.Meta_Data_df['time_point'] == str(t))]
        if self.df_checker.empty == False:
        
            ImageForNucMask = self.IMG_FOR_NUC_MASK()

            if ImageForNucMask.ndim ==2:

                nuc_bndry, nuc_mask = self.ImageAnalyzer.neuceli_segmenter(ImageForNucMask, self.Meta_Data_df["PixPerMic"].iloc[0])
                labeled_nuc, number_nuc = label(nuc_mask)
                nuc_labels = np.unique(labeled_nuc)
                if nuc_labels.max()>0:
                    dist_img = distance_transform_edt(nuc_mask)
                    dist_props = regionprops_table(labeled_nuc, dist_img, properties=('label', 'max_intensity'))
                    radial_dist_df = pd.DataFrame(dist_props)

                    nuc_centroid_locations = ndimage.measurements.center_of_mass(nuc_mask, labeled_nuc, 
                                                                                 nuc_labels[nuc_labels>0])
                    data = { "Experiment":[self.experiment_name]*number_nuc,
                             "column": [col]*number_nuc, "row": [row]*number_nuc, 
                             "time_point": [t]*number_nuc, "field_index": [fov]*number_nuc,
                             "z_slice": ["max_project"]*number_nuc, "action_index":[ai]*number_nuc}
                    df = pd.DataFrame(data)
                    regions = regionprops(labeled_nuc, ImageForNucMask)
                    props = regionprops_table(labeled_nuc, ImageForNucMask, properties=(
                                            'label', 'centroid', 'orientation', 'major_axis_length', 'minor_axis_length',
                                            'area', 'max_intensity', 'min_intensity', 'mean_intensity',
                                            'orientation', 'perimeter', 'solidity'))
                    pixpermicron = np.asarray(self.Meta_Data_df["PixPerMic"].iloc[0]).astype(float)
                    props_df = pd.DataFrame(props)
                    
#                     props_df.columns[8]=("centroid_x", "centroid_y")
#                     props_df.columns[7]= ["cell_index"]
                    props_df['major_axis_length'] = pixpermicron*props_df['major_axis_length']
                    props_df['minor_axis_length'] = pixpermicron*props_df['minor_axis_length']
                    props_df['area'] = pixpermicron*pixpermicron*props_df['area']
                    props_df['perimeter'] = pixpermicron*props_df['perimeter']
                    image_cells_df = pd.concat([df,props_df], axis=1)
                    self.cell_df = pd.concat([self.cell_df, image_cells_df], axis=0, ignore_index=True)
                    
#                     self.cell_df.columns=["Experiment","column","row","time_point","field_index","z_slice","action_index",
#                                       "cell_index", ("centroid_x", "centroid_y"),'orientation', 'major_axis_length', 
#                                       'minor_axis_length','area', 'max_intensity', 'min_intensity', 'mean_intensity',
#                                       'orientation', 'perimeter', 'solidity']


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
                                if labeled_nuc.max()>0:
                                    ch1_radial = self.RADIAL_DIST_CALC(ch1_xyz_round,ch1_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch1_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch1_num_spots,
                                         "column": [col]*ch1_num_spots, "row": [row]*ch1_num_spots, 
                                         "time_point": [t]*ch1_num_spots, "field_index": [fov]*ch1_num_spots,
                                         "z_slice": ["max_project"]*ch1_num_spots, "channel": [1]*ch1_num_spots,
                                         "action_index": [ai]*ch1_num_spots, "cell_index": ch1_spot_nuc_labels,
                                         "x_location": ch1_xyz[:,0], "y_location": ch1_xyz[:,1],
                                         "z_location": ch1_xyz[:,2], 
                                         "radial_distance":ch1_radial
                                       }
                                df_ch1 = pd.DataFrame(data)

                                cell_indices = np.unique(ch1_spot_nuc_labels)
                                if self.ImageAnalyzer.spot_params_dict["Ch1"][4]>0:
                                    for ind in cell_indices:
                                        temp_pd_df = df_ch1.loc[df_ch1["cell_index"] == ind]
                                        if temp_pd_df.__len__()!= self.ImageAnalyzer.spot_params_dict["Ch1"][4]:

                                            df_ch1 = df_ch1[df_ch1.cell_index != ind]

                                self.ch1_spot_df=pd.concat([self.ch1_spot_df,df_ch1],ignore_index=True)

                        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
                            if ch2_xyz!=[]:
                                ch2_xyz_round = np.round(np.asarray(ch2_xyz)).astype('int')
                                ch2_spot_nuc_labels = labeled_nuc[ch2_xyz_round[:,0], ch2_xyz_round[:,1]]
                                ch2_num_spots = ch2_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch2_radial = self.RADIAL_DIST_CALC(ch2_xyz_round,ch2_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch2_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch2_num_spots,
                                         "column": [col]*ch2_num_spots, "row": [row]*ch2_num_spots, 
                                         "time_point": [t]*ch2_num_spots, "field_index": [fov]*ch2_num_spots,
                                         "z_slice": ["max_project"]*ch2_num_spots, "channel": [2]*ch2_num_spots,
                                         "action_index": [ai]*ch2_num_spots, "cell_index": ch2_spot_nuc_labels,
                                         "x_location": ch2_xyz[:,0], "y_location": ch2_xyz[:,1],
                                         "z_location": ch2_xyz[:,2],
                                         "radial_distance":ch2_radial
                                       }
                                df_ch2 = pd.DataFrame(data)

                                cell_indices = np.unique(ch2_spot_nuc_labels)
                                if self.ImageAnalyzer.spot_params_dict["Ch2"][4]>0:
                                    for ind in cell_indices:
                                        temp_pd_df = df_ch2.loc[df_ch2["cell_index"] == ind]
                                        if temp_pd_df.__len__()!= self.ImageAnalyzer.spot_params_dict["Ch2"][4]:

                                            df_ch2 = df_ch2[df_ch2.cell_index != ind]

                                self.ch2_spot_df=pd.concat([self.ch2_spot_df,df_ch2],ignore_index=True)

                        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
                            if ch3_xyz!=[]:

                                ch3_xyz_round = np.round(np.asarray(ch3_xyz)).astype('int')
                                ch3_spot_nuc_labels = labeled_nuc[ch3_xyz_round[:,0], ch3_xyz_round[:,1]]
                                ch3_num_spots = ch3_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch3_radial = self.RADIAL_DIST_CALC(ch3_xyz_round,ch3_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch3_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch3_num_spots,
                                         "column": [col]*ch3_num_spots, "row": [row]*ch3_num_spots, 
                                         "time_point": [t]*ch3_num_spots, "field_index": [fov]*ch3_num_spots,
                                         "z_slice": ["max_project"]*ch3_num_spots, "channel": [3]*ch3_num_spots,
                                         "action_index": [ai]*ch3_num_spots, "cell_index": ch3_spot_nuc_labels,
                                         "x_location": ch3_xyz[:,0], "y_location": ch3_xyz[:,1],
                                         "z_location": ch3_xyz[:,2],
                                         "radial_distance": ch3_radial
                                       }

                                df_ch3 = pd.DataFrame(data)

                                cell_indices = np.unique(ch3_spot_nuc_labels)
                                if self.ImageAnalyzer.spot_params_dict["Ch3"][4]>0:
                                    for ind in cell_indices:
                                        temp_pd_df = df_ch3.loc[df_ch3["cell_index"] == ind]
                                        if temp_pd_df.__len__()!= self.ImageAnalyzer.spot_params_dict["Ch3"][4]:

                                            df_ch3 = df_ch3[df_ch3.cell_index != ind]

                                self.ch3_spot_df=pd.concat([self.ch3_spot_df, df_ch3],ignore_index=True)

                        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
                            if ch4_xyz!=[]:

                                ch4_xyz_round = np.round(np.asarray(ch4_xyz)).astype('int')
                                ch4_spot_nuc_labels = labeled_nuc[ch4_xyz_round[:,0], ch4_xyz_round[:,1]]
                                ch4_num_spots = ch4_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch4_radial = self.RADIAL_DIST_CALC(ch4_xyz_round,ch4_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch4_radial=np.nan
                                data = { "Experiment":[self.experiment_name]*ch4_num_spots,
                                         "column": [col]*ch4_num_spots, "row": [row]*ch4_num_spots, 
                                         "time_point": [t]*ch4_num_spots, "field_index": [fov]*ch4_num_spots,
                                         "z_slice": ["max_project"]*ch4_num_spots, "channel": [4]*ch4_num_spots,
                                         "action_index": [ai]*ch4_num_spots, "cell_index": ch4_spot_nuc_labels,
                                         "x_location": ch4_xyz[:,0], "y_location": ch4_xyz[:,1],
                                         "z_location": ch4_xyz[:,2],
                                         "radial_distance":ch4_radial
                                       }

                                df_ch4 = pd.DataFrame(data)

                                cell_indices = np.unique(ch4_spot_nuc_labels)
                                if self.ImageAnalyzer.spot_params_dict["Ch4"][4]>0:
                                    for ind in cell_indices:
                                        temp_pd_df = df_ch4.loc[df_ch4["cell_index"] == ind]
                                        if temp_pd_df.__len__()!= self.ImageAnalyzer.spot_params_dict["Ch4"][4]:

                                            df_ch4 = df_ch4[df_ch4.cell_index != ind]

                                self.ch4_spot_df=pd.concat([self.ch4_spot_df, df_ch4],ignore_index=True)


                        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
                            if ch5_xyz!=[]:

                                ch5_xyz_round = np.round(np.asarray(ch5_xyz)).astype('int')
                                ch5_spot_nuc_labels = labeled_nuc[ch5_xyz_round[:,0], ch5_xyz_round[:,1]]
                                ch5_num_spots = ch5_xyz.__len__()
                                if labeled_nuc.max()>0:
                                    ch5_radial = self.RADIAL_DIST_CALC(ch5_xyz_round,ch5_spot_nuc_labels,
                                                                       radial_dist_df, dist_img)
                                else:
                                    ch5_radial=np.nan


                                data = { "Experiment":[self.experiment_name]*ch5_num_spots,
                                         "column": [col]*ch5_num_spots, "row": [row]*ch5_num_spots, 
                                         "time_point": [t]*ch5_num_spots, "field_index": [fov]*ch5_num_spots,
                                         "z_slice": ["max_project"]*ch5_num_spots, "channel": [5]*ch5_num_spots,
                                         "action_index": [ai]*ch5_num_spots, "cell_index": ch5_spot_nuc_labels,
                                         "x_location": ch5_xyz[:,0], "y_location": ch5_xyz[:,1],
                                         "z_location": ch5_xyz[:,2],
                                         "radial_distance": ch5_radial
                                       }

                                df_ch5 = pd.DataFrame(data)

                                cell_indices = np.unique(ch5_spot_nuc_labels)
                                if self.ImageAnalyzer.spot_params_dict["Ch5"][4]>0:
                                    for ind in cell_indices:
                                        temp_pd_df = df_ch5.loc[df_ch5["cell_index"] == ind]
                                        if temp_pd_df.__len__()!= self.ImageAnalyzer.spot_params_dict["Ch5"][4]:

                                            df_ch5 = df_ch5[df_ch5.cell_index != ind]

                                self.ch5_spot_df=pd.concat([self.ch5_spot_df, df_ch5],ignore_index=True) 

    def Calculate_Spot_Distances(self, row, col):
    
        select_cell = self.cell_df.loc[(self.cell_df['column'] == col) & (self.cell_df['row'] == row)]    

        spot_pd_dict = {}

        if self.ch1_spot_df.empty == False:

            cell_spots_ch1 = self.ch1_spot_df.loc[(self.ch1_spot_df['column'] == col) & (self.ch1_spot_df['row'] == row)] 

            spot_pd_dict['ch1'] = cell_spots_ch1

        if self.ch2_spot_df.empty == False:

            cell_spots_ch2 = self.ch2_spot_df.loc[(self.ch2_spot_df['column'] == col) & (self.ch2_spot_df['row'] == row)] 

            spot_pd_dict['ch2'] = cell_spots_ch2

        if self.ch3_spot_df.empty == False:

            cell_spots_ch3 = self.ch3_spot_df.loc[(self.ch3_spot_df['column'] == col) & (self.ch3_spot_df['row'] == row) ] 

            spot_pd_dict['ch3'] = cell_spots_ch3

        if self.ch4_spot_df.empty == False:

            cell_spots_ch4 = self.ch4_spot_df.loc[(self.ch4_spot_df['column'] == col) & (self.ch4_spot_df['row'] == row)] 

            spot_pd_dict['ch4'] = cell_spots_ch4

        if self.ch5_spot_df.empty == False:

            cell_spots_ch5 = self.ch5_spot_df.loc[(self.ch5_spot_df['column'] == col) & (self.ch5_spot_df['row'] == row)] 

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

                    dist_pd = self.DISTANCE_calculator(spot_pd_dict,key1,key2,select_cell, row, col)
                    well_spot_dist_folder = os.path.join(self.output_folder, 'well_spots_distances')
                    if os.path.isdir(well_spot_dist_folder) == False:
                        os.mkdir(well_spot_dist_folder)

                    spot_dist_filename = 'SpotDistances_' + key1 +'_' +key2 +'_' + WELL_PLATE_ROWS[row-1] + str(col) + r'.csv'
                    spot_dist_well_csv_full_name = os.path.join(well_spot_dist_folder, spot_dist_filename)
                    dist_pd.to_csv(path_or_buf=spot_dist_well_csv_full_name, encoding='utf8')
                    print(spot_dist_filename)

    def DISTANCE_calculator(self, spot_pd_dict, key1, key2,select_cell, row, col):
    
        fovs = np.unique(np.asarray(select_cell['field_index'], dtype=int))
        time_points = np.unique(np.asarray(select_cell['time_point'], dtype=int))
        actionindices = np.unique(np.asarray(select_cell['action_index'], dtype=int))
        z_slices = np.unique(np.asarray(select_cell['z_slice']))
        dist_pd = pd.DataFrame()
        for f in fovs:
            for t in time_points:
                for a in actionindices:
                    for z in z_slices:

                        cells_in_field = select_cell.loc[ 
                                              (select_cell['time_point'] == t) & 
                                              (select_cell['field_index'] == f)& 
                                              (select_cell['action_index'] == a)&
                                              (select_cell['z_slice'] == z)] 
                        cells_in_field = np.unique(np.asarray(select_cell['label'], dtype=int))

                        for c in cells_in_field:
                            loci1 = spot_pd_dict[key1].loc[ 
                                                  (spot_pd_dict[key1]['time_point'] == t) & 
                                                  (spot_pd_dict[key1]['field_index'] == f)& 
                                                  (spot_pd_dict[key1]['action_index'] == a)&
                                                  (spot_pd_dict[key1]['z_slice'] == z) &
                                                  (spot_pd_dict[key1]['cell_index'] == c)] 
                            loci2 = spot_pd_dict[key2].loc[ 
                                                  (spot_pd_dict[key2]['time_point'] == t) & 
                                                  (spot_pd_dict[key2]['field_index'] == f)& 
                                                  (spot_pd_dict[key2]['action_index'] == a)&
                                                  (spot_pd_dict[key2]['z_slice'] == z) &
                                                  (spot_pd_dict[key2]['cell_index'] == c)] 

                            for ind1, locus1 in loci1.iterrows():
                                for ind2, locus2 in loci2.iterrows():
                                    if ind1!=ind2:
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
                                        data = {"Experiment":[self.experiment_name],
                                                'column':col, 'row': row, 'time_point': [t], 'field_index': [f], 
                                                'action_index': [a], 'z_slice':[z], 'cell_index': [c],
                                                str(s1): [ind1], str(s2): [ind2], 
                                                'XY-Distance(pixels)': [dist_2d], 'XYZ-Distance(pixels)': [dist_3d],
                                                'XY-Distance(micron)': [dist_2d*np.asarray(self.Meta_Data_df["PixPerMic"].iloc[0]).astype(float)], 
                                                'XYZ-Distance(micron)':[dist_3d*np.asarray(self.Meta_Data_df["PixPerMic"].iloc[0]).astype(float)]}
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
            ch1_xyz, ch1_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch1')
                
        if self.AnalysisGui.SpotCh2CheckBox.isChecked() == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '2')]
            ch2_xyz, ch2_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch2')
                
        if self.AnalysisGui.SpotCh3CheckBox.isChecked() == True:
            
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '3')]
            ch3_xyz, ch3_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch3')
                    
        if self.AnalysisGui.SpotCh4CheckBox.isChecked() == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '4')]
            ch4_xyz, ch4_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch4')
        
        if self.AnalysisGui.SpotCh5CheckBox.isChecked() == True:
             
            imgforspot = self.df_checker.loc[(self.df_checker['channel'] == '5')]
            ch5_xyz, ch5_xyz_3D = self.XYZ_SPOT_COORDINATES( imgforspot, ImageForNucMask, 'Ch5')
            
        return ch1_xyz, ch1_xyz_3D, ch2_xyz, ch2_xyz_3D, ch3_xyz, ch3_xyz_3D, ch4_xyz, ch4_xyz_3D, ch5_xyz, ch5_xyz_3D

    
    def XYZ_SPOT_COORDINATES(self, images_pd_df, ImageForNucMask, spot_channel):
        
        z_imglist = []
        coordinates_list = np.ones((0,3), dtype='float')
        xyz_coordinates = []
        coordinates_stack =[]
        
            
        for index, row in images_pd_df.iterrows():
            im = mpimg.imread(row['ImageName'])
            
            _z_coordinates1 = np.asarray(row['z_slice']).astype('float')
            im_uint8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            z_imglist.append( np.asarray(im_uint8))
            
            coordinates, final_spots = self.ImageAnalyzer.SpotDetector(im_uint8, self.AnalysisGui, ImageForNucMask, spot_channel)
            
            if coordinates.__len__()>0:
                _z_coordinates = np.ones((coordinates.__len__(),1), dtype='float')*_z_coordinates1
            else:
                coordinates=np.ones((0,2), dtype='float')
                _z_coordinates=np.ones((0,1), dtype='float')
            
            xyz_3d_coordinates = np.append(np.asarray(coordinates).astype('float'), _z_coordinates, 1)
            coordinates_list = np.append(coordinates_list, xyz_3d_coordinates,0)
            

        if z_imglist.__len__()>0:
            print(row['ImageName'])
            image_stack = np.stack(z_imglist, axis=2)
            max_project = image_stack.max(axis=2)
            max_im_uint8 = cv2.normalize(max_project, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            coordinates_max_project, final_spots = self.ImageAnalyzer.SpotDetector(max_im_uint8, self.AnalysisGui, 
                                                                                   ImageForNucMask, spot_channel)
        
            if coordinates_max_project.__len__()>0:

                coordinates_max_project_round = np.round(np.asarray(coordinates_max_project)).astype('int')

                coordinates_max_project = np.array(coordinates_max_project)[coordinates_max_project_round.min(axis=1)>=0,:].tolist()
                coordinates_max_project_round = coordinates_max_project_round[coordinates_max_project_round.min(axis=1)>=0,:]
                spots_z_slices = np.argmax(image_stack[coordinates_max_project_round[:,0],coordinates_max_project_round[:,1],:], axis=1)
                spots_z_coordinates = np.zeros((spots_z_slices.__len__(),1), dtype='float')
                
                for i in range(spots_z_slices.__len__()):

                    spots_z_coordinates[i] = np.asarray(images_pd_df.loc[images_pd_df['z_slice']== str(spots_z_slices[i]+1)]
                                                     ['z_coordinate'].iloc[0]).astype('float')
                if coordinates_max_project==[]:
                    coordinates_max_project==np.ones((0,2), dtype='float')
                xyz_coordinates = np.append(np.asarray(coordinates_max_project).astype('float'), spots_z_coordinates, 1)

        return xyz_coordinates, coordinates_stack
    
    
    
    
    def RADIAL_DIST_CALC(self, xyz_round, spot_nuc_labels, radial_dist_df, dist_img):
        radial_dist=[]
        eps=0.000001
        for i in range(xyz_round.__len__()):
            
            sp_dist = dist_img[xyz_round[i,0], xyz_round[i,1]]
            spot_lbl =np.int(spot_nuc_labels[i])
            if spot_lbl>0:
                cell_max = radial_dist_df.loc[radial_dist_df['label']==spot_lbl]['max_intensity'].iloc[0]
                sp_radial_dist= (cell_max-sp_dist)/(cell_max-1+eps)
            else:
                sp_radial_dist = np.nan
            radial_dist.append(sp_radial_dist)
    
        return np.array(radial_dist).astype(float)

    
        
        