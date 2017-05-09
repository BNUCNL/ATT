# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
    
import math
import numpy as np
import os
import csv
from ATT.iofunc import iofiles
from ATT.algorithm import surf_tools
from ATT.algorithm import hcp_tools


HCP_TEST_DATA = 'e:\\projects\\genetic_imaging\\HCPdata\\HCP900\\'
data_out_file = 'E:\\projects\\genetic_imaging\\HCPdata\\data\\HCPExtracted\\'
 
def motion_RMS(path_list):
    relative_meanRMS= []
    for i in path_list:
        with open(i+'\\tfMRI_WM_LR\\Movement_RelativeRMS_mean.txt','r') as f:
            relative_LR = f.readline()
        with open(i+'\\tfMRI_WM_RL\\Movement_RelativeRMS_mean.txt','r') as f:
            relative_RL = f.readline()
        relative_meanRMS.append((float(relative_LR)+float(relative_RL)/2))
    return relative_meanRMS

def motion_FD(path_list):
    rela_fd_mean = []
    
    def cal_relaFD(re1,re2):
        """
        """
        relative_fd = 0
        for i in range(0,3):
            relative_fd += abs(float(re1[i])-float(re2[i]))
        for i in range(3,6):
            relative_fd += 50*math.pi/150*(abs(float(re1[i])-float(re2[i])))
       
        return math.sqrt(relative_fd)

    def cal_relaFD_mean(ls):
        """
        """
        rela_fd = []
        for i in range(1,len(ls)):
            rela_fd.append(cal_relaFD(regre_LR[i],regre_LR[i-1]))
        indi_fd_mean = np.mean(np.array(rela_fd))
        return indi_fd_mean
    
    for i in path_list:
        with open(i+'\\tfMRI_WM_LR\\Movement_Regressors.txt','r') as f:
            s = f.readlines()
            regre_LR = np.array([j.rstrip().split() for j in s])
        with open(i+'\\tfMRI_WM_RL\\Movement_Regressors.txt','r') as f:
            s = f.readlines()
            regre_RL = np.array([j.rstrip().split() for j in s])
        indi_FD_mean = (cal_relaFD_mean(regre_LR)+cal_relaFD_mean(regre_RL))/2
        rela_fd_mean.append(indi_FD_mean) 
        
    return rela_fd_mean

def get_brainsize(path_list):
    size_list = []
    for i in path_list:
        with open(i,'r') as f:
            s = f.readlines()[23][-20:-7]
        size_list.append(float(s))
    return size_list    

class get_hcp_data(object):
    def __init__(self,stem_path):
        self.stem_path = stem_path
        
    def __get_subid(self):
        subid = os.listdir(self.stem_path)
        return subid

    def get_file_path_list(self,file_type):
        """
        generate a list containing needed file path
        file_type: func
                   stru
                   other
        """
        subid = self.__get_subid()
        
        if file_type == 'func':
            catagory = ['tfMRI_WM_hp200_s12_level2.feat','','','','','','']
            catagory = int(input('''
    please input the number of  catagoty you want:
    1.*.hp200_s12_level12.feat 
    2.*.hp200_s2_level2.feat 
    3.*.hp200_s2_level2_MSMALL.feat
    4.*.hp200_s4_level2.feat
    5.*.hp200_s4_level2_MSMALL.feat
    6.**.hp200_s8_level2.feat
    --
            '''))
            if catagory == 1:
                self.catagory = 'hp200_s12_level12.feat'
                FUNC_STEM_PATH= 'MNINonLinear\\Results\\tfMRI_WM\\tfMRI_WM_hp200_s12_level2.feat\\'
            elif catagory == 2:
                self.catagory = 'hp200_s2_level2.feat'
                FUNC_STEM_PATH= 'MNINonLinear\\Results\\tfMRI_WM\\tfMRI_WM_hp200_s2_level2.feat\\'
            elif catagory == 3:
                self.catagory = 'hp200_s2_level2_MSMALL.feat'
                FUNC_STEM_PATH= 'MNINonLinear\\Results\\tfMRI_WM\\tfMRI_WM_hp200_s2_level2_MSMALL.feat\\'
            elif catagory == 4:
                self.catagory = 'hp200_s4_level2.feat'
                FUNC_STEM_PATH= 'MNINonLinear\\Results\\tfMRI_WM\\tfMRI_WM_hp200_s4_level2.feat\\' 
            elif catagory == 5:
                self.catagory = 'hp200_s4_level2_MSMALL.feat'
                FUNC_STEM_PATH= 'MNINonLinear\\Results\\tfMRI_WM\\tfMRI_WM_hp200_s4_level2_MSMALL.feat\\'
            elif catagory == 6:
                self.catagory = 'hp200_s8_level2.feat'
                FUNC_STEM_PATH= 'MNINonLinear\\Results\\tfMRI_WM\\tfMRI_WM_hp200_s8_level2.feat\\'
            
            data_type = int(input('please input the number of your data type:\n:1.t value 2.beta\n--'))
            contrast_type = int(input('''
            please input the contrast type
            1.body-avg
            2.face-avg
            3.place-avg
            4.tool-avg
            '''))
            if data_type == 1:
                
                if  contrast_type == 1:
                    FILE_NAME = 'GrayordinatesStats\\cope19.feat\\tstat1.dtseries.nii'
                elif  contrast_type == 2:
                    FILE_NAME = 'GrayordinatesStats\\cope20.feat\\tstat1.dtseries.nii'
                elif  contrast_type == 3:
                    FILE_NAME = 'GrayordinatesStats\\cope21.feat\\tstat1.dtseries.nii'
                elif  contrast_type == 4:
                    FILE_NAME = 'GrayordinatesStats\\cope22.feat\\tstat1.dtseries.nii'
                path_list = [HCP_TEST_DATA+i+'\\'+FUNC_STEM_PATH+ FILE_NAME for i in subid]
              
            if data_type == 2:
                if  contrast_type == 1:
                    FILE_NAME = 'GrayordinatesStats\\cope19.feat\\cope1.dtseries.nii'
                elif  contrast_type == 2:
                    FILE_NAME = 'GrayordinatesStats\\cope20.feat\\cope1.dtseries.nii'
                elif  contrast_type == 3:
                    FILE_NAME = 'GrayordinatesStats\\cope21.feat\\cope1.dtseries.nii'
                elif  contrast_type == 4:
                    FILE_NAME = 'GrayordinatesStats\\cope22.feat\\cope1.dtseries.nii'
                path_list = [self.stem_path+i+'\\'+FUNC_STEM_PATH+ FILE_NAME for i in subid]
        elif file_type == 'stru':
            self.catagory = ''
    #        stru_type = ['*.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii',
    #        '*.curvature_MSMAll.32k_fs_LR.dscalar.nii','*.thickness_MSMAll.32k_fs_LR.dscalar.nii']
            stru_type_num = int(input('''
    please input the structure file you need (not number) 
    1.*.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii
    2.*.curvature_MSMAll.32k_fs_LR.dscalar.nii
    3.*.thickness_MSMAll.32k_fs_LR.dscalar.nii
    --
            '''))
            if stru_type_num == 1 :
                stru_type = '.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii'
            if stru_type_num == 2 :
                stru_type = '.curvature_MSMAll.32k_fs_LR.dscalar.nii'
            if stru_type_num == 3 :
                stru_type = '.thickness_MSMAll.32k_fs_LR.dscalar.nii'
            brain_region_num = int(input('''
    please input number of your target region
    1. left hemisphere
    2. right hemishpere
    3. whole brain
    --
             '''))
            if brain_region_num ==3:
                path_list  = [self.stem_path+i+'\\'+'MNINonLinear\\fsaverage_LR32k\\'+i+stru_type for i in subid]
            if brain_region_num ==1:
                path_list  = [self.stem_path+i+'\\'+'MNINonLinear\\fsaverage_LR32k\\'+i+'.L'+stru_type for i in subid]
            if brain_region_num ==2:
                path_list  = [self.stem_path+i+'\\'+'MNINonLinear\\fsaverage_LR32k\\'+i+'.R'+stru_type for i in subid]
                
        elif file_type == 'other':
            other_type = int(input('''
            please input the type of data you want
            1.Motion
            2.brain_size
            '''))
            if other_type == 1:
                self.other_type ='motion'
                path_list = [self.stem_path+i+'\\'+'MNINonLinear\\Results' for i in subid]
            elif other_type == 2:
                self.other_type = 'brain_size'
                path_list = [self.stem_path+i+'\\'+'T1w\\'+i+'\\'+'stats\\aseg.stats' for i in subid]
        else:
            raise Exception('please input the right file type: func, stru, other')
        return path_list
    
    def getsave_certain_data(self,file_type,label_data,output,output_path):
        '''
        '''
        path_list = self.get_file_path_list(file_type)
        data_list = []
        if file_type == 'func' or file_type=='stru':
            for i in path_list:
                data = iofiles.CIFTI(i).read_cifti()
                data_list.append(surf_tools.get_signals(data,label_data))#get roi mean value of each roi of each subject
            out_file_name = output_path+file_type+'\\'+self.catagory+'\\'+output+'.csv'
        elif file_type == 'other':
            data_list = []
            if self.other_type == 'motion':
                data_list.append(hcp_tools.motion_RMS(path_list))
                data_list.append(hcp_tools.motion_FD(path_list))
            elif self.other_type == 'brain_size':
                data_list.append(hcp_tools.get_brainsize(path_list))
                
            out_file_name = output_path+file_type+'\\'+output+'.csv'
        with open(out_file_name,'w',newline = '') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(data_list)
if __name__=='__main__':
    labelpath = 'E:\projects\genetic_imaging\HCPdata\VanEssenMap\HCP_PhaseTwo\Q1-Q6_RelatedParcellation210\MNINonLinear\\fsaverage_LR32k\\Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'

    labeldata =  iofiles.CIFTI(labelpath).read_cifti()
    getdata = get_hcp_data(HCP_TEST_DATA)
    #pathlist = getdata.get_file_path_list('func')
    roidata = getdata.getsave_certain_data('other',labeldata,'test1',data_out_file)
