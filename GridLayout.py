from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
import numpy as np
from PyQt5.QtCore import pyqtSlot

WELL_PLATE_LENGTH = 24
WELL_PLLTE_WIDTH = 16
WELL_PLATE_ROWS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

class gridgenerator(QWidget):
    
    def __init__(self, centralwidget):
        super().__init__(centralwidget)
#         self.gridLayout_centralwidget = gridLayout_centralwidget
        self.tableWidget = QtWidgets.QTableWidget(centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(10, 250, 376, 178))
#         self.gridLayout_centralwidget.addWidget(self.tableWidget, 5, 1, 6, 7)
        self.tableWidget.setBaseSize(QtCore.QSize(10, 10))
        font = QtGui.QFont()
        font.setPointSize(6)
        font.setBold(False)
        font.setWeight(50)
        self.tableWidget.setFont(font)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(WELL_PLATE_LENGTH)
        self.tableWidget.setRowCount(WELL_PLLTE_WIDTH)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
#         self.tableWidget.itemClicked.connect(self.on_click)

        for i in range(WELL_PLLTE_WIDTH):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setVerticalHeaderItem(i, item)

        for i in range(WELL_PLATE_LENGTH):
            item = QtWidgets.QTableWidgetItem()
            self.tableWidget.setHorizontalHeaderItem(i, item)

        self.tableWidget.horizontalHeader().setDefaultSectionSize(15)
        self.tableWidget.verticalHeader().setDefaultSectionSize(10)
        ### fov list
        self.FOVlist = QtWidgets.QListWidget(centralwidget)
        self.FOVlist.setGeometry(QtCore.QRect(390, 270, 45, 141))
#         self.gridLayout_centralwidget.addWidget(self.FOVlist, 6, 8, 5, 1)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.FOVlist.setFont(font)
        self.FOVlist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        #self.FOVlist.setGridSize(QtCore.QSize(8, 15))
        self.FOVlist.setBatchSize(80)
        self.FOVlist.setObjectName("FOVlist")
        self.label = QtWidgets.QLabel(centralwidget)
        self.label.setGeometry(QtCore.QRect(390, 250, 31, 16))
#         self.gridLayout_centralwidget.addWidget(self.label, 5, 8, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("FOVabel")

        ### Zlist 
        self.Zlist = QtWidgets.QListWidget(centralwidget)
        self.Zlist.setGeometry(QtCore.QRect(440, 270, 45, 141))
#         self.gridLayout_centralwidget.addWidget(self.Zlist, 6, 9, 4, 1)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Zlist.setFont(font)
        self.Zlist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        #self.Zlist.setGridSize(QtCore.QSize(8, 15))
        self.Zlist.setBatchSize(80)
        self.Zlist.setObjectName("Zlist")
        
        
        
        self.Zlabel = QtWidgets.QLabel(centralwidget)
        self.Zlabel.setGeometry(QtCore.QRect(445, 250, 16, 20))
#         self.gridLayout_centralwidget.addWidget(self.Zlabel, 5, 9, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Zlabel.setFont(font)
        self.Zlabel.setObjectName("Zlabel")
        
        ### Time list 
        self.Timelist = QtWidgets.QListWidget(centralwidget)
        self.Timelist.setGeometry(QtCore.QRect(490, 270, 45, 141))
#         self.gridLayout_centralwidget.addWidget(self.Timelist, 6, 10, 4, 1)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.Timelist.setFont(font)
        self.Timelist.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
  #      self.Timelist.setGridSize(QtCore.QSize(8, 15))
        self.Timelist.setBatchSize(80)
        self.Timelist.setObjectName("Timelist")
        
        
        
        self.Timelabel = QtWidgets.QLabel(centralwidget)
        self.Timelabel.setGeometry(QtCore.QRect(495, 250, 30, 20))
#         self.gridLayout_centralwidget.addWidget(self.Timelabel, 5, 10, 1, 1)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.Timelabel.setFont(font)
        self.Timelabel.setObjectName("Timelabel")


    
        _translate = QtCore.QCoreApplication.translate
        for i in range(WELL_PLATE_ROWS.__len__()):
            item = self.tableWidget.verticalHeaderItem(i)
            item.setText(_translate("MainWindow", WELL_PLATE_ROWS[i]))
            
        for i in range(WELL_PLATE_LENGTH):
            item = self.tableWidget.horizontalHeaderItem(i)
            item.setText(_translate("MainWindow", str(i+1)))

        self.label.setText(_translate("MainWindow", "FOV"))
        self.Zlabel.setText(_translate("MainWindow", "Z"))
        self.Timelabel.setText(_translate("MainWindow", "Time"))
    
    def GRID_INITIALIZER(self, out_df, displaygui, inout_resource_gui, ImDisplay):
        
        ### initailize well plate grid
        cols_in_use = np.unique(np.asarray(out_df['column'], dtype=int))
        rows_in_use = np.unique(np.asarray(out_df['row'], dtype=int))
        
        for c in cols_in_use:
            for r in rows_in_use:
                
                df_checker = out_df.loc[(out_df['column'] == str(c)) & (out_df['row'] == str(r))]
                
                if df_checker.empty == False:
                    self.tableWidget.setItem(r-1, c-1, QtWidgets.QTableWidgetItem())
                    self.tableWidget.item(r-1, c-1).setBackground(QtGui.QColor(10,200,10))
        current_row = rows_in_use[0]-1
        current_col = cols_in_use[0]-1
        self.tableWidget.setCurrentCell(current_row, current_col)
        self.on_click_table( out_df, displaygui, inout_resource_gui, ImDisplay)
        
    @pyqtSlot()
    def on_click_table(self, out_df, displaygui, inout_resource_gui, ImDisplay):    
        
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            img_row = currentQTableWidgetItem.row() + 1
            img_col = currentQTableWidgetItem.column() + 1
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        self.Timelist.clear()
        self.FOVlist.clear()
        self.Zlist.clear()
        #### initalizae FOV and Z list
        
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        z_values = np.unique(np.asarray(df_checker['z_slice'], dtype=int))
        for i in range(z_values.__len__()):
           
            item = QtWidgets.QListWidgetItem()
            self.Zlist.addItem(item)
        
        _translate = QtCore.QCoreApplication.translate
        __sortingEnabled = self.Zlist.isSortingEnabled()
        self.Zlist.setSortingEnabled(False)
        for i in range(z_values.__len__()):
            item = self.Zlist.item(i)
            item.setText(_translate("MainWindow", str(i+1)))
            
            if i==0:
                self.Zlist.setCurrentItem(item)
              
        self.Zlist.setSortingEnabled(__sortingEnabled)

        ### Initialize FOV List
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        fov_values = np.unique(np.asarray(df_checker['field_index'], dtype=int))
            
        for i in range(fov_values.__len__()):
            
            item = QtWidgets.QListWidgetItem()
            self.FOVlist.addItem(item)

        _translate = QtCore.QCoreApplication.translate
        __sortingEnabled = self.FOVlist.isSortingEnabled()
        self.FOVlist.setSortingEnabled(False)
        for i in range(fov_values.__len__()):
            item = self.FOVlist.item(i)
            item.setText(_translate("MainWindow", str(i+1)))
            
            if i==0:
                self.FOVlist.setCurrentItem(item)
        
        self.FOVlist.setSortingEnabled(__sortingEnabled)
        ### Initialize Time List
        df_checker = out_df.loc[(out_df['column'] == str(img_col)) & (out_df['row'] == str(img_row))]
        time_values = np.unique(np.asarray(df_checker['time_point'], dtype=int))
            
        for i in range(time_values.__len__()):
            
            item = QtWidgets.QListWidgetItem()
            self.Timelist.addItem(item)

        _translate = QtCore.QCoreApplication.translate
        __sortingEnabled = self.Timelist.isSortingEnabled()
        self.Timelist.setSortingEnabled(False)
        for i in range(time_values.__len__()):
            item = self.Timelist.item(i)
            item.setText(_translate("MainWindow", str(i+1)))
            
            if i==0:
                self.Timelist.setCurrentItem(item)
        
        self.Timelist.setSortingEnabled(__sortingEnabled)
        
        
        ImDisplay.grid_data[0] = img_col
        ImDisplay.grid_data[1] = img_row
        ImDisplay.grid_data[2] = 1
        ImDisplay.grid_data[3] = 1
        ImDisplay.grid_data[4] = 1
        ImDisplay.GET_IMAGE_NAME(displaygui)
        
    def on_click_list(self, ImDisplay, displaygui):
        
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            img_row = currentQTableWidgetItem.row() + 1
            img_col = currentQTableWidgetItem.column() + 1
            
        
        ImDisplay.grid_data[0] = img_col
        ImDisplay.grid_data[1] = img_row
        ImDisplay.grid_data[2] = self.Timelist.currentRow() + 1
        ImDisplay.grid_data[3] = self.FOVlist.currentRow() + 1
        ImDisplay.grid_data[4] = self.Zlist.currentRow() + 1
        ImDisplay.GET_IMAGE_NAME(displaygui)
            
            
            
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

