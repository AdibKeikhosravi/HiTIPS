from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import pandas as pd
import numpy as np
from distutils import util

if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

class analyzer(QWidget):

    def __init__(self, centralwidget):
        super().__init__(centralwidget)
#         self.gridLayout_centralwidget = gridLayout_centralwidget
        
        self.AnalysisLbl = QtWidgets.QLabel(centralwidget)
        self.AnalysisLbl.setGeometry(QtCore.QRect(730, 50, 120, 30))
#         self.gridLayout_centralwidget.addWidget(self.AnalysisLbl, 1, 17, 1, 3)
        font = QtGui.QFont()
        font.setFamily(".Farah PUA")
        font.setPointSize(20)
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        font.setKerning(False)
        self.AnalysisLbl.setFont(font)
        self.AnalysisLbl.setObjectName("AnalysisLbl")
        
        ##### APPLY, RESET, CLOSE BUTTONS
#         self.progressBar = QtWidgets.QProgressBar(centralwidget)
#         self.progressBar.setGeometry(QtCore.QRect(770, 800, 201, 20))
# #         self.gridLayout_centralwidget.addWidget(self.progressBar, 24, 18, 1, 5)
#         self.progressBar.setProperty("value", 24)
#         self.progressBar.setObjectName("progressBar")
        
        self.RunAnalysis = QtWidgets.QPushButton(centralwidget)
        self.RunAnalysis.setGeometry(QtCore.QRect(850, 850, 100, 32))
#         self.gridLayout_centralwidget.addWidget(self.RunAnalysis, 23, 18, 1, 2)
        self.RunAnalysis.setObjectName("RunAnalysis")
        
        self.ResetButton = QtWidgets.QPushButton(centralwidget)
        self.ResetButton.setGeometry(QtCore.QRect(730, 850, 100, 32))
#         self.gridLayout_centralwidget.addWidget(self.ResetButton, 23, 21, 1, 2)
        self.ResetButton.setObjectName("ResetButton")
        
#         self.CloseButton = QtWidgets.QPushButton(centralwidget)
#         self.CloseButton.setGeometry(QtCore.QRect(610, 770, 100, 32))
#         self.CloseButton.setObjectName("CloseButton")

        
        self.AnalysisMode = QtWidgets.QToolBox(centralwidget)
        self.AnalysisMode.setGeometry(QtCore.QRect(590, 90, 381, 370))
#         self.gridLayout_centralwidget.addWidget(self.AnalysisMode, 3, 15, 10, 15)
        self.gridLayout_AnalysisMode = QtWidgets.QGridLayout(self.AnalysisMode)
        self.gridLayout_AnalysisMode.setObjectName("gridLayout_AnalysisMode")
        
        font = QtGui.QFont()
        font.setPointSize(14)
        self.AnalysisMode.setFont(font)
        self.AnalysisMode.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.AnalysisMode.setFrameShadow(QtWidgets.QFrame.Plain)
        self.AnalysisMode.setObjectName("AnalysisMode")
        self.NucleiDetection = QtWidgets.QWidget()
        self.gridLayout_NucleiDetection = QtWidgets.QGridLayout(self.NucleiDetection)
        self.gridLayout_NucleiDetection.setObjectName("gridLayout_NucleiDetection")
#         self.NucleiDetection.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.NucleiDetection, 0, 0, 1, 1)
        ####################################
        ####### Nuclei Detection
        self.NucleiDetection.setObjectName("NucleiDetection")
        self.NucleiChLbl = QtWidgets.QLabel(self.NucleiDetection)
#         self.NucleiChLbl.setGeometry(QtCore.QRect(10, 0, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucleiChLbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucleiChLbl.setFont(font)
        self.NucleiChLbl.setObjectName("NucleiChLbl")
#         self.CellTypeLabel = QtWidgets.QLabel(self.NucleiDetection)
# #         self.CellTypeLabel.setGeometry(QtCore.QRect(10, 30, 71, 31))
#         self.gridLayout_NucleiDetection.addWidget(self.CellTypeLabel, 0, 2, 1, 1)
#         font = QtGui.QFont()
#         font.setPointSize(14)
#         self.CellTypeLabel.setFont(font)
#         self.CellTypeLabel.setObjectName("CellTypeLabel")
        self.NucDetectMethodLbl = QtWidgets.QLabel(self.NucleiDetection)
        self.NucDetectMethodLbl.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucDetectMethodLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucDetectMethodLbl.setFont(font)
        self.NucDetectMethodLbl.setObjectName("NucDetectMethodLbl")
        self.NucleiChannel = QtWidgets.QComboBox(self.NucleiDetection)
#         self.NucleiChannel.setGeometry(QtCore.QRect(80, 0, 211, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucleiChannel, 0, 1, 1, 1)
        self.NucleiChannel.setObjectName("NucleiChannel")
        self.NucleiChannel.addItem("Channel 1")
        self.NucleiChannel.addItem("Channel 2")
        self.NucleiChannel.addItem("Channel 3")
        self.NucleiChannel.addItem("Channel 4")

        self.NucDetectMethod = QtWidgets.QComboBox(self.NucleiDetection)
#         self.NucDetectMethod.setGeometry(QtCore.QRect(80, 60, 211, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucDetectMethod, 1, 1, 1, 1)
        self.NucDetectMethod.setObjectName("NucDetectMethod")
        self.NucDetectMethod.addItem("ImageProc")
        self.NucMaxZprojectCheckBox = QtWidgets.QCheckBox(self.NucleiDetection)
#         self.NucMaxZprojectCheckBox.setGeometry(QtCore.QRect(156, 100, 151, 20))
        self.gridLayout_NucleiDetection.addWidget(self.NucMaxZprojectCheckBox, 1, 2, 1, 1)
        self.NucMaxZprojectCheckBox.setObjectName("NucMaxZprojectCheckBox")
        self.NucMaxZprojectCheckBox.setChecked(True)
        self.NucFirstThreshLbl = QtWidgets.QLabel(self.NucleiDetection)
      #  self.NucFirstThreshLbl.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucFirstThreshLbl, 2, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucFirstThreshLbl.setFont(font)
        self.NucFirstThreshLbl.setObjectName("NucFirstThreshLbl")
        
        self.NucSecondThresholdSlider = QtWidgets.QSlider(self.NucleiDetection)
#         self.ThresholdSlider.setGeometry(QtCore.QRect(126, 65 , 181, 22))
        self.gridLayout_NucleiDetection.addWidget(self.NucSecondThresholdSlider, 2, 1, 1, 1)
        self.NucSecondThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.NucSecondThresholdSlider.setObjectName("NucSecondThresholdSlider")
        self.NucSecondThresholdSlider.setMaximum(100)
        self.NucSecondThresholdSlider.setMinimum(0)
        self.NucSecondThresholdSlider.setValue(42)
        self.NucSecondThresholdSlider.valueChanged.connect(lambda: self.SECOND_THRESH_LABEL_UPDATE())
        self.NucSecondThreshSliderValue = QtWidgets.QLabel(self.NucleiDetection)
#         self.NucFirstThreshSliderValue.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucSecondThreshSliderValue, 2, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucSecondThreshSliderValue.setFont(font)
        self.NucSecondThreshSliderValue.setObjectName("NucSecondThreshSliderValue")
        self.NucSecondThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucSecondThresholdSlider.value()))) 

        
        self.NucFirstThresholdSlider = QtWidgets.QSlider(self.NucleiDetection)
#         self.ThresholdSlider.setGeometry(QtCore.QRect(126, 65 , 181, 22))
        self.gridLayout_NucleiDetection.addWidget(self.NucFirstThresholdSlider, 3, 1, 1, 1)
        self.NucFirstThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.NucFirstThresholdSlider.setObjectName("NucFirstThresholdSlider")
        self.NucFirstThresholdSlider.setMaximum(100)
        self.NucFirstThresholdSlider.setMinimum(0)
        self.NucFirstThresholdSlider.setValue(39)
        self.NucFirstThresholdSlider.sliderMoved.connect(lambda: self.FIRST_THRESH_LABEL_UPDATE())
        
        self.NucFirstThreshSliderValue = QtWidgets.QLabel(self.NucleiDetection)
#         self.NucFirstThreshSliderValue.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucFirstThreshSliderValue, 3, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucFirstThreshSliderValue.setFont(font)
        self.NucFirstThreshSliderValue.setObjectName("NucFirstThreshSliderValue")
        self.NucFirstThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucFirstThresholdSlider.value()))) 
        
        self.NucSecondThreshLbl = QtWidgets.QLabel(self.NucleiDetection)
       # self.NucSecondThreshLbl.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucSecondThreshLbl, 3, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucSecondThreshLbl.setFont(font)
        self.NucSecondThreshLbl.setObjectName("NucSecondThreshLbl")
        
                
        
        self.NucleiareaLbl = QtWidgets.QLabel(self.NucleiDetection)
#         self.NucleiareaLbl.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucleiareaLbl, 4, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucleiareaLbl.setFont(font)
        self.NucleiareaLbl.setObjectName("NucleiareaLbl")
        
        self.NucleiAreaSlider = QtWidgets.QSlider(self.NucleiDetection)
#         self.ThresholdSlider.setGeometry(QtCore.QRect(126, 65 , 181, 22))
        self.gridLayout_NucleiDetection.addWidget(self.NucleiAreaSlider, 4, 1, 1, 1)
        self.NucleiAreaSlider.setOrientation(QtCore.Qt.Horizontal)
        self.NucleiAreaSlider.setObjectName("NucSecondThresholdSlider")
        self.NucleiAreaSlider.setMaximum(100)
        self.NucleiAreaSlider.setMinimum(0)
        self.NucleiAreaSlider.setValue(30)
        self.NucleiAreaSlider.valueChanged.connect(lambda: self.NUCLEI_AREA_LABEL_UPDATE())
        
        self.NucleiAreaSliderValue = QtWidgets.QLabel(self.NucleiDetection)
#         self.NucFirstThreshSliderValue.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_NucleiDetection.addWidget(self.NucleiAreaSliderValue, 4, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NucleiAreaSliderValue.setFont(font)
        self.NucleiAreaSliderValue.setObjectName("NucleiAreaSliderValue")
        self.NucleiAreaSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucleiAreaSlider.value()))) 

        self.AnalysisMode.addItem(self.NucleiDetection, "")
        
        #### Cell Boundary GUI
        self.CellBoundary = QtWidgets.QWidget()
#         self.CellBoundary.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.CellBoundary, 1, 0, 1, 1)
        self.gridLayout_CellBoundary = QtWidgets.QGridLayout(self.CellBoundary)
        self.gridLayout_CellBoundary.setObjectName("gridLayout_CellBoundary")
        
        self.CellBoundary.setObjectName("CellBoundary")
        self.CytoChLbl = QtWidgets.QLabel(self.CellBoundary)
#         self.CytoChLbl.setGeometry(QtCore.QRect(20, 0, 61, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoChLbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.CytoChLbl.setFont(font)
        self.CytoChLbl.setObjectName("CytoChLbl")
        self.CytoCellTypeLbl = QtWidgets.QLabel(self.CellBoundary)
#         self.CytoCellTypeLbl.setGeometry(QtCore.QRect(20, 30, 61, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoCellTypeLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.CytoCellTypeLbl.setFont(font)
        self.CytoCellTypeLbl.setObjectName("CytoCellTypeLbl")
        self.CytoDetectMethodLbl = QtWidgets.QLabel(self.CellBoundary)
#         self.CytoDetectMethodLbl.setGeometry(QtCore.QRect(20, 60, 61, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoDetectMethodLbl, 2, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.CytoDetectMethodLbl.setFont(font)
        self.CytoDetectMethodLbl.setObjectName("CytoDetectMethodLbl")
        self.CytoChannel = QtWidgets.QComboBox(self.CellBoundary)
#         self.CytoChannel.setGeometry(QtCore.QRect(90, 0, 211, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoChannel, 0, 1, 1, 2)
        
        self.CytoChannel.setObjectName("CytoChannel")
        self.CytoChannel.addItem("")
        self.CytoChannel.addItem("")
        self.CytoChannel.addItem("")
        self.CytoChannel.addItem("")
        self.CytoDetectMethod = QtWidgets.QComboBox(self.CellBoundary)
#         self.CytoDetectMethod.setGeometry(QtCore.QRect(90, 60, 211, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoDetectMethod, 2, 1, 1, 2)
        self.CytoDetectMethod.setObjectName("CytoDetectMethod")
        self.CytoDetectMethod.addItem("")
        self.CytoCellType = QtWidgets.QComboBox(self.CellBoundary)
#         self.CytoCellType.setGeometry(QtCore.QRect(90, 30, 211, 31))
        self.gridLayout_CellBoundary.addWidget(self.CytoCellType, 1, 1, 1, 2)
        self.CytoCellType.setObjectName("CytoCellType")
        self.CytoCellType.addItem("")
        
        
        self.AnalysisMode.addItem(self.CellBoundary, "")
        #############################################################################
        #### Spot Detection 
        self.SpotDetection = QtWidgets.QWidget()
#         self.SpotDetection.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.SpotDetection, 2, 0, 1, 1)
        self.gridLayout_SpotDetection = QtWidgets.QGridLayout(self.SpotDetection)
        self.gridLayout_SpotDetection.setObjectName("gridLayout_SpotDetection")
        self.SpotDetection.setObjectName("SpotDetection")
#         shift = 70

        self.SpotCh1CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
#         self.SpotCh1CheckBox.setGeometry(QtCore.QRect(10 + shift, 10, 51, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotCh1CheckBox, 0, 1, 1, 1)
        self.SpotCh1CheckBox.setObjectName("Ch1CheckBox")
        self.SpotCh1CheckBox.setStyleSheet("color: gray")
        
        self.SpotPerCh1SpinBox = QtWidgets.QSpinBox(self.SpotDetection)
#         self.SpotPerCh1SpinBox.setGeometry(QtCore.QRect(10 + shift, 35, 51, 24))
        self.gridLayout_SpotDetection.addWidget(self.SpotPerCh1SpinBox, 1, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.SpotPerCh1SpinBox.setFont(font)
        self.SpotPerCh1SpinBox.setObjectName("SpotPerCh1SpinBox")
        self.SpotPerCh1SpinBox.setStyleSheet("color: gray")
        
        self.SpotCh2CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
#         self.SpotCh2CheckBox.setGeometry(QtCore.QRect(70 + shift, 10, 51, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotCh2CheckBox, 0, 2, 1, 1)
        self.SpotCh2CheckBox.setObjectName("Ch2CheckBox")
        self.SpotCh2CheckBox.setStyleSheet("color: red")
        
        self.SpotPerCh2SpinBox = QtWidgets.QSpinBox(self.SpotDetection)
#         self.SpotPerCh2SpinBox.setGeometry(QtCore.QRect(70 + shift, 35, 51, 24))
        self.gridLayout_SpotDetection.addWidget(self.SpotPerCh2SpinBox, 1, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.SpotPerCh2SpinBox.setFont(font)
        self.SpotPerCh2SpinBox.setObjectName("SpotPerCh2SpinBox")
        self.SpotPerCh2SpinBox.setStyleSheet("color: red")
        
        self.SpotCh3CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
#         self.SpotCh3CheckBox.setGeometry(QtCore.QRect(130 + shift, 10, 51, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotCh3CheckBox, 0, 3, 1, 1)
        self.SpotCh3CheckBox.setObjectName("Ch3CheckBox")
        self.SpotCh3CheckBox.setStyleSheet("color: green")
        
        self.SpotPerCh3SpinBox = QtWidgets.QSpinBox(self.SpotDetection)
#         self.SpotPerCh3SpinBox.setGeometry(QtCore.QRect(130 + shift, 35, 51, 24))
        self.gridLayout_SpotDetection.addWidget(self.SpotPerCh3SpinBox, 1, 3, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.SpotPerCh3SpinBox.setFont(font)
        self.SpotPerCh3SpinBox.setObjectName("SpotPerCh3SpinBox")
        self.SpotPerCh3SpinBox.setStyleSheet("color: green")
        
        self.SpotCh4CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
#         self.SpotCh4CheckBox.setGeometry(QtCore.QRect(190 + shift, 10, 51, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotCh4CheckBox, 0, 4, 1, 1)
        self.SpotCh4CheckBox.setObjectName("Ch4CheckBox")
        self.SpotCh4CheckBox.setStyleSheet("color: blue")
        
        self.SpotPerCh4SpinBox = QtWidgets.QSpinBox(self.SpotDetection)
#         self.SpotPerCh4SpinBox.setGeometry(QtCore.QRect(190 + shift, 35, 51, 24))
        self.gridLayout_SpotDetection.addWidget(self.SpotPerCh4SpinBox, 1, 4, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.SpotPerCh4SpinBox.setFont(font)
        self.SpotPerCh4SpinBox.setObjectName("SpotPerCh4SpinBox")
        self.SpotPerCh4SpinBox.setStyleSheet("color: blue")
        
        self.SpotCh5CheckBox = QtWidgets.QCheckBox(self.SpotDetection)
#         self.SpotCh5CheckBox.setGeometry(QtCore.QRect(250 + shift, 10, 51, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotCh5CheckBox, 0, 5, 1, 1)
        self.SpotCh5CheckBox.setObjectName("Ch5CheckBox")
        self.SpotCh5CheckBox.setStyleSheet("color: orange")
        
        self.SpotPerCh5SpinBox = QtWidgets.QSpinBox(self.SpotDetection)
#         self.SpotPerCh5SpinBox.setGeometry(QtCore.QRect(250 + shift, 35, 51, 24))
        self.gridLayout_SpotDetection.addWidget(self.SpotPerCh5SpinBox, 1, 5, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.SpotPerCh5SpinBox.setFont(font)
        self.SpotPerCh5SpinBox.setObjectName("SpotPerCh5SpinBox")
        self.SpotPerCh5SpinBox.setStyleSheet("color: orange")
        
        self.SpotperchannelLbl = QtWidgets.QLabel(self.SpotDetection)
#         self.SpotperchannelLbl.setGeometry(QtCore.QRect(3, 35, 80, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotperchannelLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        
        self.SpotLocationLbl = QtWidgets.QLabel(self.SpotDetection)
#         self.SpotLocationLbl.setGeometry(QtCore.QRect(3, 75, 80, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotLocationLbl, 2, 0, 1, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.SpotLocationCbox = QtWidgets.QComboBox(self.SpotDetection)
#         self.SpotLocationCbox.setGeometry(QtCore.QRect(90, 70, 211, 31))
        self.gridLayout_SpotDetection.addWidget(self.SpotLocationCbox, 2, 2, 1, 3)
        self.SpotLocationCbox.setObjectName("SpotLocationCbox")
        self.SpotLocationCbox.addItem("Center Of Mass")
        self.SpotLocationCbox.addItem("Max Intensity")
        self.SpotLocationCbox.addItem("Centroid")
       
        
        self.SpotMaxZProject = QtWidgets.QCheckBox(self.SpotDetection)
        self.SpotMaxZProject.setChecked(True)
#         self.SpotMaxZProject.setGeometry(QtCore.QRect(160, 110, 131, 20))
        self.gridLayout_SpotDetection.addWidget(self.SpotMaxZProject, 3, 0, 1, 3)
        self.SpotMaxZProject.setObjectName("SpotMaxZProject")
        self.AnalysisMode.addItem(self.SpotDetection, "")
        #######################################################
        #### Spot Analysis
        self.SpotAnalysis = QtWidgets.QWidget()
#         self.SpotAnalysis.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.SpotAnalysis, 3, 0, 1, 1)
        self.gridLayout_SpotAnalysis = QtWidgets.QGridLayout(self.SpotAnalysis)
        self.SpotAnalysis.setObjectName("gridLayout_SpotAnalysis")

        self.SpotAnalysis.setObjectName("SpotAnalysis")
        
        self.spotanalysismethodLbl = QtWidgets.QLabel(self.SpotAnalysis)
#         self.spotanalysismethodLbl.setGeometry(QtCore.QRect(5, 0, 121, 31))
        self.gridLayout_SpotAnalysis.addWidget(self.spotanalysismethodLbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.spotanalysismethodLbl.setFont(font)
        self.spotanalysismethodLbl.setObjectName("spotanalysismethodLbl")
        self.thresholdmethodLbl = QtWidgets.QLabel(self.SpotAnalysis)
#         self.thresholdmethodLbl.setGeometry(QtCore.QRect(5, 30, 121, 31))
        self.gridLayout_SpotAnalysis.addWidget(self.thresholdmethodLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.thresholdmethodLbl.setFont(font)
        self.thresholdmethodLbl.setObjectName("thresholdmethodLbl")
        self.thresholdvalueLbl = QtWidgets.QLabel(self.SpotAnalysis)
#         self.thresholdvalueLbl.setGeometry(QtCore.QRect(5, 60, 121, 31))
        self.gridLayout_SpotAnalysis.addWidget(self.thresholdvalueLbl, 2, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.thresholdvalueLbl.setFont(font)
        self.thresholdvalueLbl.setObjectName("thresholdvalueLbl")
        
        self.spotanalysismethod = QtWidgets.QComboBox(self.SpotAnalysis)
#         self.spotanalysismethod.setGeometry(QtCore.QRect(130, 0, 211, 31))
        self.gridLayout_SpotAnalysis.addWidget(self.spotanalysismethod, 0, 1, 1, 1)
        self.spotanalysismethod.setObjectName("NucleiChannel")
        self.spotanalysismethod.addItem("LOG")
        self.spotanalysismethod.addItem("Gaussian")
        
        
        self.thresholdmethod = QtWidgets.QComboBox(self.SpotAnalysis)
#         self.thresholdmethod.setGeometry(QtCore.QRect(130, 30, 211, 31))
        self.gridLayout_SpotAnalysis.addWidget(self.thresholdmethod, 1, 1, 1, 1)
        self.thresholdmethod.setObjectName("thresholdmethod")
        self.thresholdmethod.addItem("Auto")
        self.thresholdmethod.addItem("Manual")
        
        self.ThresholdSlider = QtWidgets.QSlider(self.SpotAnalysis)
#         self.ThresholdSlider.setGeometry(QtCore.QRect(126, 65 , 181, 22))
        self.gridLayout_SpotAnalysis.addWidget(self.ThresholdSlider, 2, 1, 1, 1)
        self.ThresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ThresholdSlider.setObjectName("ThresholdSlider")
        self.ThresholdSlider.valueChanged.connect(lambda: self.SPOT_THRESH_LABEL_UPDATE())

        self.SpotThreshSliderValue = QtWidgets.QLabel(self.SpotAnalysis)
#         self.NucFirstThreshSliderValue.setGeometry(QtCore.QRect(10, 60, 61, 31))
        self.gridLayout_SpotAnalysis.addWidget(self.SpotThreshSliderValue, 2, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.SpotThreshSliderValue.setFont(font)
        self.SpotThreshSliderValue.setObjectName("SpotThreshSliderValue")
        self.SpotThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.ThresholdSlider.value()))) 
        
        
        self.ThresholdSlider.setMaximum(100)
        self.ThresholdSlider.setMinimum(0)
        self.ThresholdSlider.setValue(0)

        
        self.AnalysisMode.addItem(self.SpotAnalysis, "")
        
        #### Resutls
        self.Results = QtWidgets.QWidget()
#         self.Results.setGeometry(QtCore.QRect(0, 0, 381, 171))
        self.gridLayout_AnalysisMode.addWidget(self.Results, 4, 0, 1, 1)
        self.Results.setObjectName("Results")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.Results)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.NucMaskCheckBox = QtWidgets.QCheckBox(self.Results)
        self.NucMaskCheckBox.setObjectName("NucMaskCheckBox")
        self.gridLayout_3.addWidget(self.NucMaskCheckBox, 0, 0, 1, 1)
        self.SpotsDistance = QtWidgets.QCheckBox(self.Results)
        self.SpotsDistance.setObjectName("SpotsDistance")
        self.gridLayout_3.addWidget(self.SpotsDistance, 0, 1, 1, 1)
        
        self.NucInfoChkBox = QtWidgets.QCheckBox(self.Results)
        self.NucInfoChkBox.setObjectName("NucInfoChkBox")
        self.gridLayout_3.addWidget(self.NucInfoChkBox, 1, 0, 1, 1)
        self.SpotsLocation = QtWidgets.QCheckBox(self.Results)
        self.SpotsLocation.setObjectName("SpotsLocation")
        self.gridLayout_3.addWidget(self.SpotsLocation, 1, 1, 1, 1)
#         self.checkBox_13 = QtWidgets.QCheckBox(self.Results)
#         self.checkBox_13.setObjectName("checkBox_13")
#         self.gridLayout_3.addWidget(self.checkBox_13, 1, 2, 1, 1)
        self.AnalysisMode.addItem(self.Results, "")
        
        _translate = QtCore.QCoreApplication.translate
        self.AnalysisLbl.setText(_translate("MainWindow", "Analysis"))
        self.RunAnalysis.setText(_translate("MainWindow", "Run Analysis"))
        self.ResetButton.setText(_translate("MainWindow", "Reset"))
       # self.CloseButton.setText(_translate("MainWindow", "Close"))
        ### nuclei detection
        self.NucleiChLbl.setText(_translate("MainWindow", "Channel"))
#         self.CellTypeLabel.setText(_translate("MainWindow", "Cell Type"))
        self.NucDetectMethodLbl.setText(_translate("MainWindow", "Method"))
        self.NucleiChannel.setItemText(0, _translate("MainWindow", "Channel 1"))
        self.NucleiChannel.setItemText(1, _translate("MainWindow", "Channel 2"))
        self.NucleiChannel.setItemText(2, _translate("MainWindow", "Channel 3"))
        self.NucleiChannel.setItemText(3, _translate("MainWindow", "Channel 4"))
#         self.NucCellType.setItemText(0, _translate("MainWindow", "Fibroblasts"))
#         self.NucCellType.setItemText(1, _translate("MainWindow", "MCF10A"))
#         self.NucCellType.setItemText(2, _translate("MainWindow", "HCT116"))
#         self.NucCellType.setItemText(3, _translate("MainWindow", "U2OS"))
#         self.NucCellType.setItemText(4, _translate("MainWindow", "Mouse Mammary Tumor"))
        self.NucDetectMethod.setItemText(0, _translate("MainWindow", "Int.-based Processing"))
        self.NucMaxZprojectCheckBox.setText(_translate("MainWindow", "Max Z"))
        self.NucFirstThreshLbl.setText(_translate("MainWindow", "Nuclei Detection"))
        self.NucSecondThreshLbl.setText(_translate("MainWindow", "Nuclei Separation"))
        self.NucleiareaLbl.setText(_translate("MainWindow", "Nuclei Area (\u03BCm)\u00b2>"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.NucleiDetection), _translate("MainWindow", 
                                                                                                  "Nuclei Detection"))
        
        #### spot detection
        self.SpotCh1CheckBox.setText(_translate("MainWindow", "Ch1"))
        self.SpotCh2CheckBox.setText(_translate("MainWindow", "Ch2"))
        self.SpotCh3CheckBox.setText(_translate("MainWindow", "Ch3"))
        self.SpotCh4CheckBox.setText(_translate("MainWindow", "Ch4"))
        self.SpotCh5CheckBox.setText(_translate("MainWindow", "Ch5"))
#         self.Coor_CenterOfMass.setText(_translate("MainWindow", "Center of Mass"))
#         self.Coor_MaxIntensity.setText(_translate("MainWindow", "Maximum Intensity"))
#         self.Coor_SpotCentroid.setText(_translate("MainWindow", "Spot Centroid"))
        self.SpotperchannelLbl.setText(_translate("MainWindow", "Spots/CH:"))
        self.SpotMaxZProject.setText(_translate("MainWindow", "Max Z-projection"))
        self.SpotLocationLbl.setText(_translate("MainWindow", "Coordinates:"))
        
        self.SpotLocationCbox.setItemText(0, _translate("MainWindow", "CenterOfMass"))
        self.SpotLocationCbox.setItemText(1, _translate("MainWindow", "MaxIntensity"))
        self.SpotLocationCbox.setItemText(2, _translate("MainWindow", "Cnetroid"))
        
        ### spot analysis
        self.spotanalysismethodLbl.setText(_translate("MainWindow", "Detection Method"))
        self.thresholdmethodLbl.setText(_translate("MainWindow", "Threshold Method"))
        self.thresholdvalueLbl.setText(_translate("MainWindow", "Threshold Value"))
        
        self.spotanalysismethod.setItemText(0, _translate("MainWindow", "Laplacian of Gaussian"))
        self.spotanalysismethod.setItemText(1, _translate("MainWindow", "Gaussian"))
        self.thresholdmethod.setItemText(0, _translate("MainWindow", "Auto"))
        self.thresholdmethod.setItemText(1, _translate("MainWindow", "Manual"))
        
        #### cytoplasm analysis
        self.CytoChLbl.setText(_translate("MainWindow", "Channel"))
        self.CytoCellTypeLbl.setText(_translate("MainWindow", "Cell Type"))
        
        self.CytoDetectMethodLbl.setText(_translate("MainWindow", "Method"))
        self.CytoChannel.setItemText(0, _translate("MainWindow", "Channel 1"))
        self.CytoChannel.setItemText(1, _translate("MainWindow", "Channel 2"))
        self.CytoChannel.setItemText(2, _translate("MainWindow", "Channel 3"))
        self.CytoChannel.setItemText(3, _translate("MainWindow", "Channel 4"))
        self.CytoDetectMethod.setItemText(0, _translate("MainWindow", "Cell Specific"))
        self.CytoCellType.setItemText(0, _translate("MainWindow", "Human Cell"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.CellBoundary), _translate("MainWindow", "Cell Boundary"))
        
        ### results 
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.SpotDetection), _translate("MainWindow", "Spot Detection"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.SpotAnalysis), _translate("MainWindow", "Spot Analysis"))
        self.NucMaskCheckBox.setText(_translate("MainWindow", "Nuclei Mask"))
        self.SpotsDistance.setText(_translate("MainWindow", "Spots' Distances"))
        self.NucInfoChkBox.setText(_translate("MainWindow", "Nuclei Info"))
        self.SpotsLocation.setText(_translate("MainWindow", "Spots Location"))
#         self.checkBox_13.setText(_translate("MainWindow", "Spots Intensity"))
        self.AnalysisMode.setItemText(self.AnalysisMode.indexOf(self.Results), _translate("MainWindow", "Results"))
    
    
    def FIRST_THRESH_LABEL_UPDATE(self):
        
        self.NucFirstThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucFirstThresholdSlider.value())))
        
    def SECOND_THRESH_LABEL_UPDATE(self):
        
        self.NucSecondThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucSecondThresholdSlider.value())))
        
    def NUCLEI_AREA_LABEL_UPDATE(self):
        
        self.NucleiAreaSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.NucleiAreaSlider.value()))) 
    
    def SPOT_THRESH_LABEL_UPDATE(self):
        
        self.SpotThreshSliderValue.setText(QtCore.QCoreApplication.translate("MainWindow", str(self.ThresholdSlider.value())))
    
    
    def SAVE_CONFIGURATION(self, csv_filename):
        
        config_data = {
            
            "nuclei_channel": self.NucleiChannel.currentText(),
            "nuclei_detection_method": self.NucDetectMethod.currentText(),
            "nuclei_z_project":  self.NucMaxZprojectCheckBox.isChecked(),
            "nuclei_detection": self.NucSecondThresholdSlider.value(),
            "nuclei_separation": self.NucFirstThresholdSlider.value(),
            "nuclei_area": self.NucleiAreaSlider.value(),
            "ch1_spot": self.SpotCh1CheckBox.isChecked(),
            "ch2_spot": self.SpotCh2CheckBox.isChecked(),
            "ch3_spot": self.SpotCh3CheckBox.isChecked(),
            "ch4_spot": self.SpotCh4CheckBox.isChecked(),
            "ch5_spot": self.SpotCh5CheckBox.isChecked(),
            "ch1_spots/ch": self.SpotPerCh1SpinBox.value(),
            "ch2_spots/ch": self.SpotPerCh2SpinBox.value(),
            "ch3_spots/ch": self.SpotPerCh3SpinBox.value(),
            "ch4_spots/ch": self.SpotPerCh4SpinBox.value(),
            "ch5_spots/ch": self.SpotPerCh5SpinBox.value(),
            "spot_coordinates": self.SpotLocationCbox.currentText(),
            "spot_z_project": self.SpotMaxZProject.isChecked(),
            "spot_detection_method": self.spotanalysismethod.currentText(),
            "spot_threshold_method": self.thresholdmethod.currentText(),
            "spot_threshold_value": self.ThresholdSlider.value()
            
        }
        
        config_df = pd.DataFrame.from_dict(config_data, orient='index')
        
        config_df.to_csv(csv_filename)
    
    def file_save(self):
        self.fnames, _  = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File')
        self.csv_filename = self.fnames + r'.csv'
        self.SAVE_CONFIGURATION(self.csv_filename)
        
    def LOAD_CONFIGURATION(self):
        
         
        
        options = QtWidgets.QFileDialog.Options()
        self.fnames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Select Configuration File...',
                                                                '', "Configuration files (*.csv)"
                                                                , options=options)
        print(self.fnames[0])
        conf = pd.read_csv(self.fnames[0])
        
        self.NucleiChannel.setCurrentText(conf[conf['Unnamed: 0']== 'nuclei_channel']['0'].iloc[0])
        self.NucDetectMethod.setCurrentText(conf[conf['Unnamed: 0']== 'nuclei_detection_method']['0'].iloc[0])
        self.NucMaxZprojectCheckBox.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'nuclei_z_project']['0'].iloc[0])))
        self.NucSecondThresholdSlider.setValue(np.array(conf[conf['Unnamed: 0']== 'nuclei_detection']['0'].iloc[0]).astype(int))
        self.NucFirstThresholdSlider.setValue(np.array(conf[conf['Unnamed: 0']== 'nuclei_separation']['0'].iloc[0]).astype(int))
        self.NucleiAreaSlider.setValue(np.array(conf[conf['Unnamed: 0']== 'nuclei_area']['0'].iloc[0]).astype(int))
        self.SpotCh1CheckBox.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'ch1_spot']['0'].iloc[0])))
        self.SpotCh2CheckBox.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'ch2_spot']['0'].iloc[0])))
        self.SpotCh3CheckBox.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'ch3_spot']['0'].iloc[0])))
        self.SpotCh4CheckBox.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'ch4_spot']['0'].iloc[0])))
        self.SpotCh5CheckBox.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'ch5_spot']['0'].iloc[0])))
        self.SpotPerCh1SpinBox.setValue(np.array(conf[conf['Unnamed: 0']== 'ch1_spots/ch']['0'].iloc[0]).astype(int))
        self.SpotPerCh2SpinBox.setValue(np.array(conf[conf['Unnamed: 0']== 'ch2_spots/ch']['0'].iloc[0]).astype(int))
        self.SpotPerCh3SpinBox.setValue(np.array(conf[conf['Unnamed: 0']== 'ch3_spots/ch']['0'].iloc[0]).astype(int))
        self.SpotPerCh4SpinBox.setValue(np.array(conf[conf['Unnamed: 0']== 'ch4_spots/ch']['0'].iloc[0]).astype(int))
        self.SpotPerCh5SpinBox.setValue(np.array(conf[conf['Unnamed: 0']== 'ch5_spots/ch']['0'].iloc[0]).astype(int))
        self.SpotLocationCbox.setCurrentText(conf[conf['Unnamed: 0']== 'spot_coordinates']['0'].iloc[0])
        self.SpotMaxZProject.setChecked(bool(util.strtobool(conf[conf['Unnamed: 0']== 'spot_z_project']['0'].iloc[0])))
        self.spotanalysismethod.setCurrentText(conf[conf['Unnamed: 0']== 'spot_detection_method']['0'].iloc[0])
        self.thresholdmethod.setCurrentText(conf[conf['Unnamed: 0']== 'spot_threshold_method']['0'].iloc[0])
        self.ThresholdSlider.setValue(np.array(conf[conf['Unnamed: 0']== 'spot_threshold_value']['0'].iloc[0]).astype(int))

        
        (bool(util.strtobool(conf[conf['Unnamed: 0']== 'spot_z_project']['0'].iloc[0])))