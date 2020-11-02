#from controlpanel import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

IO_GUI_HEIGHT = 180
class display(QWidget):
    
    def __init__(self, centralwidget):
        super().__init__(centralwidget)
        
        
#         self.gridLayout_centralwidget = gridLayout_centralwidget
        self.viewer = PhotoViewer(self)
# # #         self.gridLayout_centralwidget.addWidget(self.viewer, 11, 1, 15, 10)

        
        self.MaxHistSlider = QtWidgets.QSlider(self)
        self.MaxHistSlider.setGeometry(QtCore.QRect(325, 600 + IO_GUI_HEIGHT, 140, 22))
# # #         self.gridLayout_centralwidget.addWidget(self.MaxHistSlider, 30, 7, 1, 3)
        self.MaxHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MaxHistSlider.setObjectName("MaxHistSlider")
        
        self.MinHistSlider = QtWidgets.QSlider(self)
        self.MinHistSlider.setGeometry(QtCore.QRect(185, 600 + IO_GUI_HEIGHT, 140, 22))
# # #         self.gridLayout_centralwidget.addWidget(self.MinHistSlider, 30, 4, 1, 3)
        self.MinHistSlider.setOrientation(QtCore.Qt.Horizontal)
        self.MinHistSlider.setObjectName("MinHistSlider")
        
                
       
        self.Ch1CheckBox = QtWidgets.QCheckBox(self)
        self.Ch1CheckBox.setGeometry(QtCore.QRect(480, 170 + IO_GUI_HEIGHT +20, 51, 20))
# #         self.gridLayout_centralwidget.addWidget(self.Ch1CheckBox, 12, 12, 1, 1)
        self.Ch1CheckBox.setObjectName("Ch1CheckBox")
        self.Ch1CheckBox.setStyleSheet("color: gray")
        self.Ch2CheckBox = QtWidgets.QCheckBox(self)
        self.Ch2CheckBox.setGeometry(QtCore.QRect(480, 190 + IO_GUI_HEIGHT +20, 51, 20))
#         self.gridLayout_centralwidget.addWidget(self.Ch2CheckBox, 13, 12, 1, 1)
        self.Ch2CheckBox.setObjectName("Ch2CheckBox")
        self.Ch2CheckBox.setStyleSheet("color: red")
        self.Ch3CheckBox = QtWidgets.QCheckBox(self)
        self.Ch3CheckBox.setGeometry(QtCore.QRect(480, 210 + IO_GUI_HEIGHT +20, 51, 20))
#         self.gridLayout_centralwidget.addWidget(self.Ch3CheckBox, 14, 12, 1, 1)
        self.Ch3CheckBox.setObjectName("Ch3CheckBox")
        self.Ch3CheckBox.setStyleSheet("color: green")
        self.Ch4CheckBox = QtWidgets.QCheckBox(self)
        self.Ch4CheckBox.setGeometry(QtCore.QRect(480, 230 + IO_GUI_HEIGHT +20, 51, 20))
#         self.gridLayout_centralwidget.addWidget(self.Ch4CheckBox, 15, 12, 1, 1)
        self.Ch4CheckBox.setObjectName("Ch4CheckBox")
        self.Ch4CheckBox.setStyleSheet("color: blue")
        self.Ch5CheckBox = QtWidgets.QCheckBox(self)
        self.Ch5CheckBox.setGeometry(QtCore.QRect(480, 250 + IO_GUI_HEIGHT +20, 51, 20))
#         self.gridLayout_centralwidget.addWidget(self.Ch5CheckBox, 16, 12, 1, 1)
        self.Ch5CheckBox.setObjectName("Ch5CheckBox")
        self.Ch5CheckBox.setStyleSheet("color: orange")
        
        self.HistChLabel = QtWidgets.QLabel(self)
        self.HistChLabel.setGeometry(QtCore.QRect(10, 600 + IO_GUI_HEIGHT, 100, 31))
#         self.gridLayout_centralwidget.addWidget(self.HistChLabel, 30, 1, 1, 2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.HistChLabel.setFont(font)
        self.HistChLabel.setObjectName("HistChLabel")
        self.HistChannel = QtWidgets.QComboBox(self)
        self.HistChannel.setGeometry(QtCore.QRect(110, 600 + IO_GUI_HEIGHT, 71, 31))
#         self.gridLayout_centralwidget.addWidget(self.HistChannel, 30, 3, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.HistChannel.setFont(font)
        self.HistChannel.setObjectName("HistChannel")
        self.HistChannel.addItem("Ch 1")
        self.HistChannel.addItem("Ch 2")
        self.HistChannel.addItem("Ch 3")
        self.HistChannel.addItem("Ch 4")
        
        
        self.NuclMaskCheckBox = QtWidgets.QCheckBox(self)
        self.NuclMaskCheckBox.setGeometry(QtCore.QRect(480, 300 + IO_GUI_HEIGHT +20, 70, 20))
#         self.gridLayout_centralwidget.addWidget(self.NuclMaskCheckBox, 18, 12, 1, 2)
        self.NuclMaskCheckBox.setObjectName("NucMaskCheckBox")
        self.NuclMaskCheckBox.setStyleSheet("color: red")
        self.NucPreviewMethod = QtWidgets.QComboBox(self)
        self.NucPreviewMethod.setGeometry(QtCore.QRect(480, 320 + IO_GUI_HEIGHT +20, 95, 31))
#         self.gridLayout_centralwidget.addWidget(self.NucPreviewMethod, 19, 12, 1, 2)
        self.NucPreviewMethod.setObjectName("NucPreviewMethod")
        self.NucPreviewMethod.addItem("Boundary")
        self.NucPreviewMethod.addItem("Area")
        
        self.SpotsCheckBox = QtWidgets.QCheckBox(self)
        self.SpotsCheckBox.setGeometry(QtCore.QRect(480, 370 + IO_GUI_HEIGHT +20, 60, 20))
#         self.gridLayout_centralwidget.addWidget(self.SpotsCheckBox, 20, 12, 1, 2)
        self.SpotsCheckBox.setObjectName("SpotDetection")
        self.SpotsCheckBox.setStyleSheet("color: green")
        
        
        self.CytoPreviewCheck = QtWidgets.QCheckBox(self)
        self.CytoPreviewCheck.setGeometry(QtCore.QRect(480, 410 + IO_GUI_HEIGHT +20, 45, 31))
#         self.gridLayout_centralwidget.addWidget(self.CytoPreviewCheck, 21, 12, 1, 2)
        self.CytoPreviewCheck.setObjectName("CytoPreviewCheck")
        self.CytoPreviewCheck.setStyleSheet("color: blue")
        
        self.CytoDisplayMethod = QtWidgets.QComboBox(self)
        self.CytoDisplayMethod.setGeometry(QtCore.QRect(480, 430 + IO_GUI_HEIGHT +20, 95, 31))
#         self.gridLayout_centralwidget.addWidget(self.CytoDisplayMethod, 22, 12, 1, 2)
        self.CytoDisplayMethod.setObjectName("CytoDisplayMethod")
        self.CytoDisplayMethod.addItem("Boundary")
        self.CytoDisplayMethod.addItem("Area")
        
        
        _translate = QtCore.QCoreApplication.translate
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
        
        self.CytoPreviewCheck.setText(_translate("MainWindow", "Cell"))
        self.CytoDisplayMethod.setItemText(0, _translate("MainWindow", "Boundary"))
        self.CytoDisplayMethod.setItemText(1, _translate("MainWindow", "Area"))

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self.setGeometry(QtCore.QRect(30, 180+ IO_GUI_HEIGHT, 440, 420))
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

