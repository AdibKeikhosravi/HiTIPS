from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import multiprocessing as mp
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


class InOut_resource(QWidget):
    Output_dir = []
    Num_CPU_cores = 0
    def __init__(self, centralwidget):
        super().__init__(centralwidget)
        
        self.tabWidget = QtWidgets.QTabWidget(centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 541, 151))
        self.tabWidget.setObjectName("tabWidget")
        self.IO = QtWidgets.QWidget()
        self.IO.setObjectName("IO")
        self.gridLayout_IO = QtWidgets.QGridLayout(self.IO)
        self.gridLayout_IO.setObjectName("gridLayout_IO")
        self.DeviceTypeLbl = QtWidgets.QLabel(self.IO)
#         self.DeviceTypeLbl.setGeometry(QtCore.QRect(10, 13, 50, 17))
        self.gridLayout_IO.addWidget(self.DeviceTypeLbl, 0, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.DeviceTypeLbl.setFont(font)
        self.DeviceTypeLbl.setObjectName("DeviceTypeLbl")
        self.DeviceType = QtWidgets.QComboBox(self.IO)
#         self.DeviceType.setGeometry(QtCore.QRect(60, 11, 110, 26))
        self.gridLayout_IO.addWidget(self.DeviceType, 0, 1, 1, 1)
        self.DeviceType.setObjectName("DeviceType")
        self.DeviceType.addItem("CellVoyager")
        self.DeviceType.addItem("")
        
        self.LoadMetadataButton = QtWidgets.QPushButton(self.IO)
#         self.LoadMetadataButton.setGeometry(QtCore.QRect(175, 9, 121, 32))
        self.gridLayout_IO.addWidget(self.LoadMetadataButton, 0, 2, 1, 1)
        self.LoadMetadataButton.setObjectName("LoadMetadataButton")
        
        self.LoadImageButton = QtWidgets.QPushButton(self.IO)
#         self.LoadImageButton.setGeometry(QtCore.QRect(300, 9, 101, 32))
        self.gridLayout_IO.addWidget(self.LoadImageButton, 0, 3, 1, 1)
        self.LoadImageButton.setObjectName("LoadImageButton")
        
        
        self.OutFldrButton = QtWidgets.QPushButton(self.IO)
#         self.OutFldrButton.setGeometry(QtCore.QRect(405, 9, 124, 32))
        self.gridLayout_IO.addWidget(self.OutFldrButton, 0, 4, 1, 1)
        self.OutFldrButton.setObjectName("OutFldrButton")
        
        self.OutFldrButton.clicked.connect(lambda: self.OUTPUT_FOLDER_LOADBTN())
        
        self.NumFilesLoadedLbl = QtWidgets.QLabel(self.IO)
#         self.NumFilesLoadedLbl.setGeometry(QtCore.QRect(10, 48, 148, 18))
        self.gridLayout_IO.addWidget(self.NumFilesLoadedLbl, 1, 0, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NumFilesLoadedLbl.setFont(font)
        self.NumFilesLoadedLbl.setObjectName("NumFilesLoadedLbl")
        self.DisplayCheckBox = QtWidgets.QCheckBox(self.IO)
#         self.DisplayCheckBox.setGeometry(QtCore.QRect(410, 49, 81, 21))
        self.gridLayout_IO.addWidget(self.DisplayCheckBox, 1, 4, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.DisplayCheckBox.setFont(font)
        self.DisplayCheckBox.setObjectName("DisplayCheckBox")
        self.DisplayCheckBox.setChecked(False)
        self.tabWidget.addTab(self.IO, "")
        self.Resources = QtWidgets.QWidget()
        self.Resources.setObjectName("Resources")
        self.gridLayout_Resource = QtWidgets.QGridLayout(self.Resources)
        self.gridLayout_Resource.setObjectName("gridLayout_Resource")
        self.CPUInquiry = QtWidgets.QPushButton(self.Resources)
#         self.CPUInquiry.setGeometry(QtCore.QRect(10, 10, 121, 21))
        self.gridLayout_Resource.addWidget(self.CPUInquiry, 0, 0, 1, 1)
        self.CPUInquiry.setObjectName("CPUInquiry")
        
        self.NumCPUAvail = QtWidgets.QLCDNumber(self.Resources)
#         self.NumCPUAvail.setGeometry(QtCore.QRect(140, 10, 51, 21))
        self.gridLayout_Resource.addWidget(self.NumCPUAvail, 0, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.NumCPUAvail.setFont(font)
        self.NumCPUAvail.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.NumCPUAvail.setObjectName("NumCPUAvail")
        
        self.CPUAvailLabel = QtWidgets.QLabel(self.Resources)
#         self.CPUAvailLabel.setGeometry(QtCore.QRect(200, 10, 211, 20))
        self.gridLayout_Resource.addWidget(self.CPUAvailLabel, 0, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.CPUAvailLabel.setFont(font)
        self.CPUAvailLabel.setObjectName("CPUAvailLabel")
        
        self.NumCPUsSpinBox = QtWidgets.QSpinBox(self.Resources)
#         self.NumCPUsSpinBox.setGeometry(QtCore.QRect(410, 10, 48, 24))
        self.gridLayout_Resource.addWidget(self.NumCPUsSpinBox, 0, 3, 1, 1)
        self.NumCPUsSpinBox.setObjectName("NumCPUsSpinBox")
        
        self.CPUInquiry.clicked.connect(lambda: self.ON_CPU_INQUIRY_BUTTON())
        
        
        self.CPUsInUseLabel = QtWidgets.QLabel(self.Resources)
#         self.CPUsInUseLabel.setGeometry(QtCore.QRect(460, 10, 61, 21))
        self.gridLayout_Resource.addWidget(self.CPUsInUseLabel, 0, 4, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.CPUsInUseLabel.setFont(font)
        self.CPUsInUseLabel.setObjectName("CPUsInUseLabel")
        
        self.GPUInquiryButton = QtWidgets.QPushButton(self.Resources)
#         self.GPUInquiryButton.setGeometry(QtCore.QRect(10, 50, 113, 21))
        self.gridLayout_Resource.addWidget(self.GPUInquiryButton, 1, 0, 1, 1)
        self.GPUInquiryButton.setObjectName("GPUInquiryButton")
        
        self.NumGPUAvail = QtWidgets.QLCDNumber(self.Resources)
#         self.NumGPUAvail.setGeometry(QtCore.QRect(130, 50, 51, 21))
        self.gridLayout_Resource.addWidget(self.NumGPUAvail, 1, 1, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.NumGPUAvail.setFont(font)
        self.NumGPUAvail.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.NumGPUAvail.setObjectName("NumGPUAvail")
        
        self.GPUAvailLabel = QtWidgets.QLabel(self.Resources)
#         self.GPUAvailLabel.setGeometry(QtCore.QRect(190, 50, 201, 20))
        self.gridLayout_Resource.addWidget(self.GPUAvailLabel, 1, 2, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.GPUAvailLabel.setFont(font)
        self.GPUAvailLabel.setObjectName("GPUAvailLabel")
        
        self.NumGPUsSpinBox = QtWidgets.QSpinBox(self.Resources)
#         self.NumGPUsSpinBox.setGeometry(QtCore.QRect(370, 50, 48, 24))
        self.gridLayout_Resource.addWidget(self.NumGPUsSpinBox, 1, 3, 1, 1)
        self.NumGPUsSpinBox.setObjectName("NumGPUsSpinBox")
        self.GPUsInUseLabel = QtWidgets.QLabel(self.Resources)
#         self.GPUsInUseLabel.setGeometry(QtCore.QRect(420, 50, 51, 21))
        self.gridLayout_Resource.addWidget(self.GPUsInUseLabel, 1, 4, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.GPUsInUseLabel.setFont(font)
        self.GPUsInUseLabel.setObjectName("GPUsInUseLabel")
        
        
        
        
        self.tabWidget.addTab(self.Resources, "")
        
        _translate = QtCore.QCoreApplication.translate
        self.DeviceTypeLbl.setText(_translate("MainWindow", "Device"))
        self.DeviceType.setItemText(0, _translate("MainWindow", "CellVoyager"))
        self.LoadMetadataButton.setText(_translate("MainWindow", "Load MetaData"))
        self.LoadImageButton.setText(_translate("MainWindow", "Load Images"))
        self.OutFldrButton.setText(_translate("MainWindow", "Output folder"))
        self.DisplayCheckBox.setText(_translate("MainWindow", "Display"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.IO), _translate("MainWindow", "Input Output"))
        self.GPUAvailLabel.setText(_translate("MainWindow", "GPUs Are Available, Use"))
        self.GPUInquiryButton.setText(_translate("MainWindow", "GPU INQUIRY"))
        self.GPUsInUseLabel.setText(_translate("MainWindow", "GPU(s)"))
        self.CPUInquiry.setText(_translate("MainWindow", "CPU Core INQUIRY"))
        self.CPUAvailLabel.setText(_translate("MainWindow", "CPU Cores Are Available, Use"))
        self.CPUsInUseLabel.setText(_translate("MainWindow", "Core(s)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Resources), _translate("MainWindow", "Available Resources"))
    
    def OUTPUT_FOLDER_LOADBTN(self):
        
        options = QtWidgets.QFileDialog.Options()
        self.Output_dir = QtWidgets.QFileDialog.getExistingDirectory(self, caption= "Select Output Directory", options=options)

    def ON_CPU_INQUIRY_BUTTON(self):
        
        self.Num_CPU_cores = mp.cpu_count()
        self.NumCPUAvail.display(self.Num_CPU_cores)