from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget

class InOut_resource(QWidget):

    def __init__(self, centralwidget):
        super().__init__(centralwidget)
        
        self.tabWidget = QtWidgets.QTabWidget(centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 541, 151))
        self.tabWidget.setObjectName("tabWidget")
        self.IO = QtWidgets.QWidget()
        self.IO.setObjectName("IO")
        self.DeviceTypeLbl = QtWidgets.QLabel(self.IO)
        self.DeviceTypeLbl.setGeometry(QtCore.QRect(10, 13, 50, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.DeviceTypeLbl.setFont(font)
        self.DeviceTypeLbl.setObjectName("DeviceTypeLbl")
        self.DeviceType = QtWidgets.QComboBox(self.IO)
        self.DeviceType.setGeometry(QtCore.QRect(60, 11, 110, 26))
        self.DeviceType.setObjectName("DeviceType")
        self.DeviceType.addItem("CellVoyager")
        self.DeviceType.addItem("")
        
        self.LoadMetadataButton = QtWidgets.QPushButton(self.IO)
        self.LoadMetadataButton.setGeometry(QtCore.QRect(175, 9, 121, 32))
        self.LoadMetadataButton.setObjectName("LoadMetadataButton")
        
        self.LoadImageButton = QtWidgets.QPushButton(self.IO)
        self.LoadImageButton.setGeometry(QtCore.QRect(300, 9, 101, 32))
        self.LoadImageButton.setObjectName("LoadImageButton")
        
        self.OutFldrButton = QtWidgets.QPushButton(self.IO)
        self.OutFldrButton.setGeometry(QtCore.QRect(405, 9, 124, 32))
        self.OutFldrButton.setObjectName("OutFldrButton")
        
        self.NumFilesLoadedLbl = QtWidgets.QLabel(self.IO)
        self.NumFilesLoadedLbl.setGeometry(QtCore.QRect(10, 48, 148, 18))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.NumFilesLoadedLbl.setFont(font)
        self.NumFilesLoadedLbl.setObjectName("NumFilesLoadedLbl")
        self.DisplayCheckBox = QtWidgets.QCheckBox(self.IO)
        self.DisplayCheckBox.setGeometry(QtCore.QRect(410, 49, 81, 21))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.DisplayCheckBox.setFont(font)
        self.DisplayCheckBox.setObjectName("DisplayCheckBox")
        self.DisplayCheckBox.setChecked(False)
        self.tabWidget.addTab(self.IO, "")
        self.Resources = QtWidgets.QWidget()
        self.Resources.setObjectName("Resources")
        self.NumGPUAvail = QtWidgets.QLCDNumber(self.Resources)
        self.NumGPUAvail.setGeometry(QtCore.QRect(130, 50, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.NumGPUAvail.setFont(font)
        self.NumGPUAvail.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.NumGPUAvail.setObjectName("NumGPUAvail")
        self.GPUAvailLabel = QtWidgets.QLabel(self.Resources)
        self.GPUAvailLabel.setGeometry(QtCore.QRect(190, 50, 201, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.GPUAvailLabel.setFont(font)
        self.GPUAvailLabel.setObjectName("GPUAvailLabel")
        self.GPUInquiryButton = QtWidgets.QPushButton(self.Resources)
        self.GPUInquiryButton.setGeometry(QtCore.QRect(10, 50, 113, 21))
        self.GPUInquiryButton.setObjectName("GPUInquiryButton")
        self.NumGPUsSpinBox = QtWidgets.QSpinBox(self.Resources)
        self.NumGPUsSpinBox.setGeometry(QtCore.QRect(370, 50, 48, 24))
        self.NumGPUsSpinBox.setObjectName("NumGPUsSpinBox")
        self.GPUsInUseLabel = QtWidgets.QLabel(self.Resources)
        self.GPUsInUseLabel.setGeometry(QtCore.QRect(420, 50, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.GPUsInUseLabel.setFont(font)
        self.GPUsInUseLabel.setObjectName("GPUsInUseLabel")
        self.CPUInquiry = QtWidgets.QPushButton(self.Resources)
        self.CPUInquiry.setGeometry(QtCore.QRect(10, 10, 121, 21))
        self.CPUInquiry.setObjectName("CPUInquiry")
        self.CPUAvailLabel = QtWidgets.QLabel(self.Resources)
        self.CPUAvailLabel.setGeometry(QtCore.QRect(200, 10, 211, 20))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.CPUAvailLabel.setFont(font)
        self.CPUAvailLabel.setObjectName("CPUAvailLabel")
        self.CPUsInUseLabel = QtWidgets.QLabel(self.Resources)
        self.CPUsInUseLabel.setGeometry(QtCore.QRect(460, 10, 61, 21))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.CPUsInUseLabel.setFont(font)
        self.CPUsInUseLabel.setObjectName("CPUsInUseLabel")
        self.NumCPUAvail = QtWidgets.QLCDNumber(self.Resources)
        self.NumCPUAvail.setGeometry(QtCore.QRect(140, 10, 51, 21))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.NumCPUAvail.setFont(font)
        self.NumCPUAvail.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.NumCPUAvail.setObjectName("NumCPUAvail")
        self.NumCPUsSpinBox = QtWidgets.QSpinBox(self.Resources)
        self.NumCPUsSpinBox.setGeometry(QtCore.QRect(410, 10, 48, 24))
        self.NumCPUsSpinBox.setObjectName("NumCPUsSpinBox")
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