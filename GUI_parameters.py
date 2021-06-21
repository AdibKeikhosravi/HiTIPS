
class Gui_Params(object):
    
    def __init__(self,analysisgui, inout_resource_gui):
        self.AnalysisGui = analysisgui
        self.inout_resource_gui = inout_resource_gui
        self.NucInfoChkBox_check_status = self.AnalysisGui.NucInfoChkBox.isChecked()
        self.SpotsLocation_check_status = self.AnalysisGui.SpotsLocation.isChecked()
        self.SpotLocationCbox_currentText = self.AnalysisGui.SpotLocationCbox.currentText()
        self.SpotsDistance_check_status = self.AnalysisGui.SpotsDistance.isChecked()
        self.NucMaskCheckBox_status_check = self.AnalysisGui.NucMaskCheckBox.isChecked()
        self.NucMaxZprojectCheckBox_status_check = self.AnalysisGui.NucMaxZprojectCheckBox.isChecked()
        self.SpotMaxZProject_status_check = self.AnalysisGui.SpotMaxZProject.isChecked()
        self.SpotCh1CheckBox_status_check = self.AnalysisGui.SpotCh1CheckBox.isChecked()
        self.SpotCh2CheckBox_status_check = self.AnalysisGui.SpotCh2CheckBox.isChecked()
        self.SpotCh3CheckBox_status_check = self.AnalysisGui.SpotCh3CheckBox.isChecked()
        self.SpotCh4CheckBox_status_check = self.AnalysisGui.SpotCh4CheckBox.isChecked()
        self.SpotCh5CheckBox_status_check = self.AnalysisGui.SpotCh5CheckBox.isChecked()
        self.NucleiChannel_index = self.AnalysisGui.NucleiChannel.currentIndex()
        self.NumCPUsSpinBox_value = self.inout_resource_gui.NumCPUsSpinBox.value()
        
        
        
        
        
        