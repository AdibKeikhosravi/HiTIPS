from PyQt5 import QtCore, QtGui, QtWidgets
import DisplayGUI, AnalysisGUI, IO_ResourceGUI, GridLayout
from PyQt5.QtWidgets import QWidget
import Display, InputOutput, MetaData_Reader
import pandas as pd
from xml.dom import minidom
import os

class ControlPanel(QWidget):
       
    #self.displaygui = QtWidgets.QGroupBox()    
    Meta_Data_df = pd.DataFrame()
    
    def controlUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(976, 822)
        font = QtGui.QFont()
        font.setItalic(True)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
######  Instantiating GUI classes

        self.inout_resource_gui = IO_ResourceGUI.InOut_resource(self.centralwidget)
        self.inputoutputcontrol = InputOutput.inputoutput_control()
        
        self.displaygui = DisplayGUI.display(self.centralwidget)
        
        
        self.analysisgui = AnalysisGUI.analyzer(self.centralwidget)
        self.ImDisplay = Display.imagedisplayer(self.analysisgui,self.centralwidget)
        self.PlateGrid = GridLayout.gridgenerator(self.centralwidget)
        
        self.CV_Reader = MetaData_Reader.CellVoyager()        
    
        MainWindow.setCentralWidget(self.centralwidget)
        
######  Input Output loader controllers

        self.inout_resource_gui.LoadMetadataButton.clicked.connect(lambda: self.ON_CLICK_LOADBUTTON())
        self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda:
                                                                     self.ImDisplay.display_initializer(self.Meta_Data_df,
                                                                     self.displaygui, self.inout_resource_gui))
        
        #self.inout_resource_gui.DisplayCheckBox.stateChanged.connect(lambda: INSTANTIATE_DISPLAY())
                                                                     
####### Display GUI controlers
        
        self.displaygui.ColScroller.sliderMoved.connect(lambda:
                                                        self.ImDisplay.COL_SCROLLER_MOVE_UPDATE(self.displaygui))
        self.displaygui.ColSpinBox.valueChanged.connect(lambda: 
                                                        self.ImDisplay.COL_SPINBOX_UPDATE(self.displaygui))
        
        self.displaygui.RowScroller.sliderMoved.connect(lambda:
                                                        self.ImDisplay.ROW_SCROLLER_MOVE_UPDATE(self.displaygui))
        
        self.displaygui.RowSpinBox.valueChanged.connect(lambda:
                                                        self.ImDisplay.ROW_SPINBOX_UPDATE(self.displaygui))
        
        self.displaygui.ZScroller.sliderMoved.connect(lambda: 
                                                      self.ImDisplay.Z_SCROLLER_MOVE_UPDATE(self.displaygui))
        
        self.displaygui.ZSpinBox.valueChanged.connect(lambda: 
                                                      self.ImDisplay.Z_SPINBOX_UPDATE(self.displaygui))
        
        self.displaygui.FOVScroller.sliderMoved.connect(lambda:
                                                        self.ImDisplay.FOV_SCROLLER_MOVE_UPDATE(self.displaygui))
        self.displaygui.FOVSpinBox.valueChanged.connect(lambda: self.ImDisplay.FOV_SPINBOX_UPDATE(self.displaygui))
        
        self.displaygui.TScroller.sliderMoved.connect(lambda: self.ImDisplay.T_SCROLLER_MOVE_UPDATE(self.displaygui))
        self.displaygui.TSpinBox.valueChanged.connect(lambda: self.ImDisplay.T_SPINBOX_UPDATE(self.displaygui))
        ###### CHANNELS CHECKBOXES
        self.displaygui.Ch1CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch2CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch3CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.Ch4CheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
         ###### histogram controllers
        self.displaygui.MaxHistSlider.valueChanged.connect(lambda:
                                                           self.ImDisplay.MAX_HIST_SLIDER_UPDATE(self.displaygui))
        
        self.displaygui.MinHistSlider.valueChanged.connect(lambda:
                                                           self.ImDisplay.MIN_HIST_SLIDER_UPDATE(self.displaygui))
        
        self.displaygui.MinHistSpinBox.valueChanged.connect(lambda:
                                                            self.ImDisplay.MIN_HIST_SPINBOX_UPDATE(self.displaygui))
        
        self.displaygui.MaxHistSpinBox.valueChanged.connect(lambda:
                                                            self.ImDisplay.MAX_HIST_SPINBOX_UPDATE(self.displaygui))
        
        ####### Nuclei and spot visualization controllers
        
        self.displaygui.NuclMaskCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        self.displaygui.NucPreviewMethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        self.displaygui.SpotsCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.displaygui.SpotPreviewMethod.currentIndexChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        ####### Analysis Gui Controllers
        
        self.analysisgui.NucMaxZprojectCheckBox.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        self.analysisgui.SpotMaxZProject.stateChanged.connect(lambda: self.ImDisplay.GET_IMAGE_NAME(self.displaygui))
        
        
        
        
        
        ##################
        
        self.retranslateUi(MainWindow)
        self.analysisgui.AnalysisMode.setCurrentIndex(3)
        self.inout_resource_gui.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    #def INSTANTIATE_DISPLAY(self):
        
    #    self.displaygui = DisplayGUI.display(self.centralwidget)
        
        

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HiTIPS"))
        
    def READ_FROM_METADATA(self, metadatafilename):
    
        PATH_TO_FILES = os.path.split(metadatafilename)[0]
        self.mydoc = minidom.parse(metadatafilename)
        self.items = self.mydoc.getElementsByTagName('bts:MeasurementRecord')
        
        df_cols = ["ImageName", "Column", "Row", "TimePoint", "FieldIndex", "ZSlice", "Channel"]
        rows = []
        
        for i in range(self.items.length):
    
            rows.append({
                
                 "ImageName": os.path.join(PATH_TO_FILES, self.items[i].firstChild.data), 
                 "Column": self.items[i].attributes['bts:Column'].value, 
                 "Row": self.items[i].attributes['bts:Row'].value, 
                 "TimePoint": self.items[i].attributes['bts:TimePoint'].value, 
                 "FieldIndex": self.items[i].attributes['bts:FieldIndex'].value, 
                 "ZSlice": self.items[i].attributes['bts:ZIndex'].value, 
                 "Channel": self.items[i].attributes['bts:Ch'].value
                })
        
        self.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
        
        metadatafilename = os.path.join(PATH_TO_FILES,'MeasurementDetail.mrf')
        mydoc = minidom.parse(metadatafilename)
        PATH_TO_FILES = os.path.split(metadatafilename)[0]
        items = mydoc.getElementsByTagName('bts:MeasurementChannel')
        
        
        PixPerMic_Text= 'Pixel Size = '"{:.2f}".format(float(items[0].attributes['bts:HorizontalPixelDimension'].value)) + '\u03BC'+'m'
        self.inout_resource_gui.NumFilesLoadedLbl.setText(QtCore.QCoreApplication.translate("MainWindow", PixPerMic_Text))
        self.inout_resource_gui.NumFilesLoadedLbl.setStyleSheet("color: blue")


        
    def ON_CLICK_LOADBUTTON(self):
        
        options = QtWidgets.QFileDialog.Options()
        self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Image Files...',
                                                                '', "Image files (*.tiff *tif  *.jpg); MLF files (*.mlf)"
                                                                , options=options)
        
        filename, file_extension = os.path.splitext(self.fnames[0])
        self.READ_FROM_METADATA(self.fnames[0])
        
    def wheelEvent(self, event):
        print('ttt')
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:                
            factor = 0.8
            self._zoom -= 1
        if self._zoom > 0:
            displaygui.ImageView.scale(factor, factor)
        elif self._zoom == 0:
            self.fitInView()
        else:
            self._zoom = 0
            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    cp = ControlPanel()
    cp.controlUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

