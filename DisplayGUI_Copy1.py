#from controlpanel import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
IO_GUI_HEIGHT = 180
class display(QWidget):
    
    def __init__(self, centralwidget):
        super().__init__(centralwidget)
        
        
        
        self.viewer = PhotoViewer(self)
        

        self.ResetHistogram = QtWidgets.QPushButton(self)
        self.ResetHistogram.setGeometry(QtCore.QRect(470, 587 + IO_GUI_HEIGHT, 71, 32))
        self.ResetHistogram.setObjectName("ResetHistogram")
        self.MaxHistSlider = QtWidgets.QSlider(self)
        self.MaxHistSlider.setGeometry(QtCore.QRect(325, 580 + IO_GUI_HEIGHT, 140, 22))
        self.MaxHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MaxHistSlider.setObjectName("MaxHistSlider")
        
        self.MinHistSlider = QtWidgets.QSlider(self)
        self.MinHistSlider.setGeometry(QtCore.QRect(185, 580 + IO_GUI_HEIGHT, 140, 22))
        self.MinHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MinHistSlider.setObjectName("MinHistSlider")
        
                
        self.HistAutoAdjust = QtWidgets.QPushButton(self)
        self.HistAutoAdjust.setGeometry(QtCore.QRect(470, 560 + IO_GUI_HEIGHT, 71, 32))
        self.HistAutoAdjust.setObjectName("HistAutoAdjust")
        self.Ch1CheckBox = QtWidgets.QCheckBox(self)
        self.Ch1CheckBox.setGeometry(QtCore.QRect(450, 170 + IO_GUI_HEIGHT, 51, 20))
        self.Ch1CheckBox.setObjectName("Ch1CheckBox")
        self.Ch1CheckBox.setStyleSheet("color: gray")
        self.Ch2CheckBox = QtWidgets.QCheckBox(self)
        self.Ch2CheckBox.setGeometry(QtCore.QRect(450, 190 + IO_GUI_HEIGHT, 51, 20))
        self.Ch2CheckBox.setObjectName("Ch2CheckBox")
        self.Ch2CheckBox.setStyleSheet("color: red")
        self.Ch3CheckBox = QtWidgets.QCheckBox(self)
        self.Ch3CheckBox.setGeometry(QtCore.QRect(450, 210 + IO_GUI_HEIGHT, 51, 20))
        self.Ch3CheckBox.setObjectName("Ch3CheckBox")
        self.Ch3CheckBox.setStyleSheet("color: green")
        self.Ch4CheckBox = QtWidgets.QCheckBox(self)
        self.Ch4CheckBox.setGeometry(QtCore.QRect(450, 230 + IO_GUI_HEIGHT, 51, 20))
        self.Ch4CheckBox.setObjectName("Ch4CheckBox")
        self.Ch4CheckBox.setStyleSheet("color: blue")
        self.Ch5CheckBox = QtWidgets.QCheckBox(self)
        self.Ch5CheckBox.setGeometry(QtCore.QRect(450, 250 + IO_GUI_HEIGHT, 51, 20))
        self.Ch5CheckBox.setObjectName("Ch5CheckBox")
        self.Ch5CheckBox.setStyleSheet("color: orange")
        
        self.HistChLabel = QtWidgets.QLabel(self)
        self.HistChLabel.setGeometry(QtCore.QRect(10, 577 + IO_GUI_HEIGHT, 100, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.HistChLabel.setFont(font)
        self.HistChLabel.setObjectName("HistChLabel")
        self.HistChannel = QtWidgets.QComboBox(self)
        self.HistChannel.setGeometry(QtCore.QRect(110, 580 + IO_GUI_HEIGHT, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.HistChannel.setFont(font)
        self.HistChannel.setObjectName("HistChannel")
        self.HistChannel.addItem("Ch 1")
        self.HistChannel.addItem("Ch 2")
        self.HistChannel.addItem("Ch 3")
        self.HistChannel.addItem("Ch 4")
        
        
        self.NuclMaskCheckBox = QtWidgets.QCheckBox(self)
        self.NuclMaskCheckBox.setGeometry(QtCore.QRect(5, 615 + IO_GUI_HEIGHT, 70, 20))
        self.NuclMaskCheckBox.setObjectName("NucMaskCheckBox")
        self.NuclMaskCheckBox.setStyleSheet("color: red")
        self.NucPreviewMethod = QtWidgets.QComboBox(self)
        self.NucPreviewMethod.setGeometry(QtCore.QRect(75, 610 + IO_GUI_HEIGHT, 110, 31))
        self.NucPreviewMethod.setObjectName("NucPreviewMethod")
        self.NucPreviewMethod.addItem("Boundary")
        self.NucPreviewMethod.addItem("Area")
        
        self.SpotsCheckBox = QtWidgets.QCheckBox(self)
        self.SpotsCheckBox.setGeometry(QtCore.QRect(205, 615 + IO_GUI_HEIGHT, 60, 20))
        self.SpotsCheckBox.setObjectName("SpotDetection")
        self.SpotsCheckBox.setStyleSheet("color: green")
        
        self.SpotPreviewMethod = QtWidgets.QComboBox(self)
        self.SpotPreviewMethod.setGeometry(QtCore.QRect(265, 610 + IO_GUI_HEIGHT, 80, 31))
        self.SpotPreviewMethod.setObjectName("SpotPreviewMethod")
        self.SpotPreviewMethod.addItem("Dots")
        self.SpotPreviewMethod.addItem("Cross")
        
        self.CytoPreviewCheck = QtWidgets.QCheckBox(self)
        self.CytoPreviewCheck.setGeometry(QtCore.QRect(365, 610 + IO_GUI_HEIGHT, 45, 31))
        self.CytoPreviewCheck.setObjectName("CytoPreviewCheck")
        self.CytoPreviewCheck.setStyleSheet("color: blue")
        
        self.CytoDisplayMethod = QtWidgets.QComboBox(self)
        self.CytoDisplayMethod.setGeometry(QtCore.QRect(410, 610 + IO_GUI_HEIGHT, 110, 31))
        self.CytoDisplayMethod.setObjectName("CytoDisplayMethod")
        self.CytoDisplayMethod.addItem("Boundary")
        self.CytoDisplayMethod.addItem("Area")
        
        
        _translate = QtCore.QCoreApplication.translate
        self.ResetHistogram.setText(_translate("MainWindow", "Reset"))
        self.HistAutoAdjust.setText(_translate("MainWindow", "Auto"))
        self.Ch1CheckBox.setText(_translate("MainWindow", "Ch1"))
        self.Ch2CheckBox.setText(_translate("MainWindow", "Ch2"))
        self.Ch3CheckBox.setText(_translate("MainWindow", "Ch3"))
        self.Ch4CheckBox.setText(_translate("MainWindow", "Ch4"))
        self.Ch5CheckBox.setText(_translate("MainWindow", "Ch5"))
        self.HistChLabel.setText(_translate("MainWindow", "Adjust Intensity"))
        self.HistChannel.setItemText(0, _translate("MainWindow", "Ch 1"))
        self.HistChannel.setItemText(1, _translate("MainWindow", "Ch 2"))
        self.HistChannel.setItemText(2, _translate("MainWindow", "Ch 3"))
        self.HistChannel.setItemText(3, _translate("MainWindow", "Ch 4"))
        self.NuclMaskCheckBox.setText(_translate("MainWindow", "Nuclei"))
        self.NucPreviewMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.NucPreviewMethod.setItemText(1, _translate("MainWindow", "Area"))
        self.SpotsCheckBox.setText(_translate("MainWindow", "Spots"))
        self.SpotPreviewMethod.setItemText(0, _translate("MainWindow", "Dots"))
        self.SpotPreviewMethod.setItemText(1, _translate("MainWindow", "Cross"))
        self.CytoPreviewCheck.setText(_translate("MainWindow", "Cell"))
        self.CytoDisplayMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.CytoDisplayMethod.setItemText(1, _translate("MainWindow", "Area"))

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self.setGeometry(QtCore.QRect(30, 220+ IO_GUI_HEIGHT, 411, 350))
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0
            
#     def fitInView(self, scale=True):
#         rect = QtCore.QRectF(self._photo.pixmap().rect())
        
#         self.setSceneRect(rect)
            
#         unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
#         self.scale(1 / unity.width(), 1 / unity.height())
#         viewrect = self.viewport().rect()
#         scenerect = self.transform().mapRect(rect)
#         factor = min(viewrect.width() / scenerect.width(),
#                      viewrect.height() / scenerect.height())
#         self.scale(factor, factor)
#         self._zoom = 0        
    

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()
    

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

