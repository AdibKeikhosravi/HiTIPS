
from xml.dom import minidom
import os
import pandas as pd
from PIL import Image
from tifffile import imsave
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import imagej
ij=imagej.init(r'V:\Users_Data\Adib\Fiji.app')
from jnius import autoclass


class Ui_MainWindow(QWidget):
    Pix_shift = 0
    Input_dir = []
    Output_dir = []
    
    
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(350, 250)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        ## Input folder button
        self.Inputfolder = QtWidgets.QPushButton(self.centralwidget)
        self.Inputfolder.setGeometry(QtCore.QRect(10, 30, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Inputfolder.setFont(font)
        self.Inputfolder.setObjectName("Inputfolder")
        
        self.Inputfolder.clicked.connect(lambda: self.INPUT_FOLDER_LOADBTN())
        
        ## Output folder button
        self.Outputfolder = QtWidgets.QPushButton(self.centralwidget)
        self.Outputfolder.setGeometry(QtCore.QRect(190, 30, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Outputfolder.setFont(font)
        self.Outputfolder.setObjectName("Outputfolder")
        
        self.Outputfolder.clicked.connect(lambda: self.OUTPUT_FOLDER_LOADBTN())

        ## pixel size spinbox
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox.setGeometry(QtCore.QRect(230, 90, 91, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.doubleSpinBox.setFont(font)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        ## pixel size label
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 90, 200, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")

        ## Run Button
        self.runbtn = QtWidgets.QPushButton(self.centralwidget)
        self.runbtn.setGeometry(QtCore.QRect(110, 150, 141, 51))
        self.runbtn.setObjectName("runbtn")
        
        self.runbtn.clicked.connect(lambda: self.ON_RUN_BTN())
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 353, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ImageStitcher"))
        self.Inputfolder.setText(_translate("MainWindow", "Input Folder"))
        self.Outputfolder.setText(_translate("MainWindow", "Output Folder"))
        self.label.setText(_translate("MainWindow", "Image Overlap In Pixels"))
        self.runbtn.setText(_translate("MainWindow", "Run!"))

        
    def INPUT_FOLDER_LOADBTN(self):
        
        options = QtWidgets.QFileDialog.Options()
        self.Input_dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption= "Select Input Directory", options=options)
        
        #### READING .MES FILE FOR EXTRACTING PIXEL OVERLAP
        for mes_file in os.listdir(self.Input_dir):
            if mes_file.endswith(".mes"):
                mesfilename = os.path.join(self.Input_dir,mes_file)
                mydoc_mes = minidom.parse(mesfilename)
                items_mes = mydoc_mes.getElementsByTagName('bts:PartialTiledPosition')

                self.Pix_shift = int(items_mes[0].attributes['bts:OverlappingPixels'].value)/2
                self.doubleSpinBox.setValue(self.Pix_shift)
    def OUTPUT_FOLDER_LOADBTN(self):
        
        options = QtWidgets.QFileDialog.Options()
        self.Output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption= "Select Output Directory", options=options)
     
     
    def ON_RUN_BTN(self):
        
        for mlf_file in os.listdir(self.Input_dir):
            if mlf_file.endswith(".mlf"):
                metadatafilename = os.path.join(self.Input_dir,mlf_file)

        mydoc = minidom.parse(metadatafilename)
        PATH_TO_FILES = os.path.split(metadatafilename)[0]
        items = mydoc.getElementsByTagName('bts:MeasurementRecord')

        #### READING .MES FILE FOR EXTRACTING PIXEL OVERLAP
        for mes_file in os.listdir(self.Input_dir):
            if mes_file.endswith(".mes"):
                mesfilename = os.path.join(self.Input_dir,mes_file)

        mydoc_mes = minidom.parse(mesfilename)
        items_mes = mydoc_mes.getElementsByTagName('bts:PartialTiledPosition')

        Pix_shift = int(items_mes[0].attributes['bts:OverlappingPixels'].value)/2

        df_cols = ["ImageName", "Column", "Row", "TimePoint", "FieldIndex", "Channel", "Xposition", "Yposition", "Zposition"]
        rows = []

        for i in range(items.length):
            if items[i].attributes['bts:Type'].value=="IMG":
                rows.append({
                             "ImageName": os.path.join(items[i].firstChild.data), 
                             "Column": items[i].attributes['bts:Column'].value, 
                             "Row": items[i].attributes['bts:Row'].value, 
                             "TimePoint": items[i].attributes['bts:TimePoint'].value, 
                             "FieldIndex": items[i].attributes['bts:FieldIndex'].value, 
                             "Channel":items[i].attributes['bts:Ch'].value,
                             "Xposition": items[i].attributes['bts:X'].value,
                             "Yposition": items[i].attributes['bts:Y'].value,
                             "Zposition": items[i].attributes['bts:Z'].value 
                            })
            
        out_df = pd.DataFrame(rows, columns = df_cols)

        xpos = out_df['Xposition'].unique()
        ypos = out_df['Yposition'].unique()
        channels_names = out_df['Channel'].unique()

        num_channels = channels_names.__len__()

        for i in range(num_channels):
                            
            configfile = open(os.path.join(self.Input_dir,"TileConfiguration.txt"),"w") 
            L = ["# Define the number of dimensions we are working on \n",
                 "dim  = 2",
                 "\n\n",
                 "# Define the image coordinates \n"]
            configfile.writelines(L)

            for x in range(xpos.__len__()):
                for y in range(ypos.__len__()):
                    select_color =out_df.loc[(out_df['Xposition'] == str(xpos[x])) 
                                             & (out_df['Yposition'] == str(ypos[y]))
                                             & (out_df['Channel'] == str(channels_names[i]))]

                    if i==0:
                        im = Image.open(os.path.join(self.Input_dir,select_color['ImageName'].iloc[0]))
                        height, width = np.asarray(im).shape
                    x_shift = x*(width-Pix_shift)
                    y_shift = y*(height-Pix_shift)
                    img_name = select_color['ImageName'].iloc[0]
                    lines = img_name + "; ; (" + str(x_shift) + "," + str(y_shift) + ") \n"
                    configfile.writelines(lines)
            configfile.close()

            args = {'type': 'Positions from file', 'order': 'Defined by TileConfiguration', 'directory':self.Input_dir, 
                    'ayout_file': 'TileConfiguration.txt', 'fusion_method': 'Linear Blending', 'regression_threshold': '0.30', 
                    'max/avg_displacement_threshold':'2.50', 'absolute_displacement_threshold': '3.50', 
                    'computation_parameters': 'Save memory (but be slower)', 'image_output': 'Fuse and display'}

            plugin = "Grid/Collection stitching"
            print(self.Input_dir)
            ij.py.run_plugin(plugin, args)
            
            WindowManager = autoclass('ij.WindowManager')
            result = WindowManager.getCurrentImage()
            result_np = ij.py.from_java(result).astype('uint16')
            stitched_image_name = 'Stitched_Channel_'+ str(channels_names[i]) + '.tif'
            Out_img_name = os.path.join(self.Output_dir, stitched_image_name)
            imsave(Out_img_name,result_np)
            os.remove(os.path.join(self.Input_dir,"TileConfiguration.txt"))
            

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

