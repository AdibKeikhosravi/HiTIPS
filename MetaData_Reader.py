from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget
from xml.dom import minidom
import numpy as np
import pandas as pd
import HiTIPS

class CellVoyager(object):
        
    def READ_FROM_METADATA(self, metadatafilename):
    
        self.mydoc = minidom.parse(metadatafilename)
        self.items = self.mydoc.getElementsByTagName('bts:MeasurementRecord')
        
        df_cols = ["ImageName", "Column", "Row", "TimePoint", "FieldIndex", "ZSlice", "Channel"]
        rows = []
        
        for i in range(self.items.length):
    
            rows.append({
                
                 "ImageName": self.items[i].firstChild.data, 
                 "Column": self.items[i].attributes['bts:Column'].value, 
                 "Row": self.items[i].attributes['bts:Row'].value, 
                 "TimePoint": self.items[i].attributes['bts:TimePoint'].value, 
                 "FieldIndex": self.items[i].attributes['bts:FieldIndex'].value, 
                 "ZSlice": self.items[i].attributes['bts:ZIndex'].value, 
                 "Channel": self.items[i].attributes['bts:Ch'].value
                })
        
        #HiTIPS.ControlPanel.Meta_Data_df = pd.DataFrame(rows, columns = df_cols)
        
        aa = pd.DataFrame(rows, columns = df_cols)
        HiTIPS.ControlPanel.RETURN_METADATA_DATAFRAME (aa)