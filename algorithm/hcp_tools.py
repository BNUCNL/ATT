# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
    
import math
import numpy as np
import os
import csv
from ATT.iofunc import iofiles
from ATT.algorithm import surf_tools
import pandas as pd
from ATT.util import decorators
from memory_profiler import profile
from memory_profiler import memory_usage

HCP_TEST_DATA = 'e:\\projects\\genetic_imaging\\HCPdata\\HCP900\\'
data_out_file = 'E:\\projects\\genetic_imaging\\HCPdata\\data\\HCPExtracted\\'
 

class get_hcp_data(object):
    def __init__(self,stem_path):
        self.stem_path = stem_path
        subid = os.listdir(stem_path)
        self.subid = subid

    def motion_FD(self,path_list):
        power_rela_fd_mean = []
        dt_rela_fd_mean = []
        def cal_relaFD(re1, re2):
            """
            """
            relative_fd = 0
            for i in range(0, 3):
                relative_fd += abs(re1[i] - re2[i])
            for i in range(3, 6):
                relative_fd += 50 * math.pi / 150 * (abs(re1[i] - re2[i]))

            return math.sqrt(relative_fd)

        def cal_relaFD_mean(ls):
            """
            """
            ls_float = [list(map(lambda x: float(x), j)) for j in ls]
            rela_fd = []
            for i in range(1, len(ls_float)):
                rela_fd.append(cal_relaFD(ls_float[i], ls_float[i - 1]))
            indi_fd_mean = np.mean(np.array(rela_fd))
            return indi_fd_mean

        for i in path_list:
            try:
                with open(i + '/tfMRI_WM_LR/Movement_Regressors.txt', 'r') as f:
                    s = f.readlines()
                    regre_LR = np.array([j.rstrip().split() for j in s])
                with open(i + '/tfMRI_WM_RL/Movement_Regressors.txt', 'r') as f:
                    s = f.readlines()
                    regre_RL = np.array([j.rstrip().split() for j in s])
                indi_FD_mean = (cal_relaFD_mean(regre_LR) + cal_relaFD_mean(regre_RL)) / 2
                power_rela_fd_mean.append(indi_FD_mean)
            except IOError as e:
                print(e)
                power_rela_fd_mean.append([])
        for i in path_list:
            try:
                with open(i + '/tfMRI_WM_LR/Movement_Regressors_dt.txt', 'r') as f:
                    s = f.readlines()
                    regre_LR = regre_LR = np.array([j.rstrip().split() for j in s])
                with open(i + '/tfMRI_WM_RL/Movement_Regressors_dt.txt', 'r') as f:
                    s = f.readlines()
                    regre_RL = np.array([j.rstrip().split() for j in s])

                indi_FD_mean = (cal_relaFD_mean(regre_LR) + cal_relaFD_mean(regre_RL)) / 2
                dt_rela_fd_mean.append(indi_FD_mean)
            except IOError as e:
                print(e)
                dt_rela_fd_mean.append([])

        return power_rela_fd_mean,dt_rela_fd_mean

    def get_brainsize(self,path_list):
        etiv_size_list = []
        for i in path_list:
            try:
                with open(i, 'r') as f:
                    s = f.readlines()[33][-21:-7]
                etiv_size_list.append(float(s))
            except IOError as e:
                print(e)
                etiv_size_list.append([])
        return etiv_size_list

    def motion_RMS(self,path_list):
        relative_meanRMS = []
        for i in path_list:
            try:
                with open(i + '\\tfMRI_WM_LR\\Movement_RelativeRMS_mean.txt', 'r') as f:
                    relative_LR = f.readline()
                with open(i + '\\tfMRI_WM_RL\\Movement_RelativeRMS_mean.txt', 'r') as f:
                    relative_RL = f.readline()
                relative_meanRMS.append((float(relative_LR) + float(relative_RL) / 2))
            except IOError as e:
                print(e)
                relative_meanRMS.append([])
        return relative_meanRMS
    def get_file_path_list(self,file_type):
        """
        generate a list containing needed file path
        file_type: func
                   stru
                   other
        """
        
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
                FUNC_STEM_PATH= 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s12_level2.feat/'
            elif catagory == 2:
                self.catagory = 'hp200_s2_level2.feat'
                FUNC_STEM_PATH= 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2.feat/'
            elif catagory == 3:
                self.catagory = 'hp200_s2_level2_MSMALL.feat'
                FUNC_STEM_PATH= 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s2_level2_MSMALL.feat/'
            elif catagory == 4:
                self.catagory = 'hp200_s4_level2.feat'
                FUNC_STEM_PATH= 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s4_level2.feat/'
            elif catagory == 5:
                self.catagory = 'hp200_s4_level2_MSMALL.feat'
                FUNC_STEM_PATH= 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s4_level2_MSMALL.feat/'
            elif catagory == 6:
                self.catagory = 'hp200_s8_level2.feat'
                FUNC_STEM_PATH= 'MNINonLinear/Results/tfMRI_WM/tfMRI_WM_hp200_s8_level2.feat/'
            
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
                    FILE_NAME = 'GrayordinatesStats/cope19.feat/tstat1.dtseries.nii'
                elif  contrast_type == 2:
                    FILE_NAME = 'GrayordinatesStats/cope20.feat/tstat1.dtseries.nii'
                elif  contrast_type == 3:
                    FILE_NAME = 'GrayordinatesStats/cope21.feat/tstat1.dtseries.nii'
                elif  contrast_type == 4:
                    FILE_NAME = 'GrayordinatesStats/cope22.feat/tstat1.dtseries.nii'
                path_list = [self.stem_path+i+'/'+FUNC_STEM_PATH+ FILE_NAME for i in self.subid]
              
            if data_type == 2:
                if  contrast_type == 1:
                    FILE_NAME = 'GrayordinatesStats/cope19.feat/cope1.dtseries.nii'
                elif  contrast_type == 2:
                    FILE_NAME = 'GrayordinatesStats/cope20.feat/cope1.dtseries.nii'
                elif  contrast_type == 3:
                    FILE_NAME = 'GrayordinatesStats/cope21.feat/cope1.dtseries.nii'
                elif  contrast_type == 4:
                    FILE_NAME = 'GrayordinatesStats/cope22.feat/cope1.dtseries.nii'
                path_list = [self.stem_path+i+'/'+FUNC_STEM_PATH+ FILE_NAME for i in self.subid]
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
                path_list  = [self.stem_path+i+'/'+'MNINonLinear/fsaverage_LR32k/'+i+stru_type for i in self.subid]
            if brain_region_num ==1:
                path_list  = [self.stem_path+i+'/'+'MNINonLinear/fsaverage_LR32k/'+i+'.L'+stru_type for i in self.subid]
            if brain_region_num ==2:
                path_list  = [self.stem_path+i+'/'+'MNINonLinear/fsaverage_LR32k/'+i+'.R'+stru_type for i in self.subid]
                
        elif file_type == 'other':
            other_type = int(input('''
            please input the type of data you want
            1.Motion
            2.brain_size
            '''))
            if other_type == 1:
                self.other_type ='motion'
                path_list = [self.stem_path+i+'/'+'MNINonLinear/Results' for i in self.subid]
            elif other_type == 2:
                self.other_type = 'brain_size'
                path_list = [self.stem_path+i+'/'+'T1w/'+i+'/'+'stats/aseg.stats' for i in self.subid]
        else:
            raise Exception('please input the right file type: func, stru, other')
        return path_list

    @decorators.timer
    #@profile
    def getsave_certain_data(self,file_type,label_data,output,output_path):
        '''
        '''
        path_list = self.get_file_path_list(file_type)
        data_list = []
        if file_type == 'func' or file_type=='stru':
            for i in path_list:
                try:
                    data = iofiles.CIFTI(i).read_cifti()
                    data_list.append(
                    surf_tools.get_signals(data, label_data))  # get roi mean value of each roi of each subject
                except IOError as e:
                    print(e)
                    data_list.append([])


            out_file_name = output_path+file_type+'/'+self.catagory+'/'+output+'.csv'
        elif file_type == 'other':
            data_list = []
            if self.other_type == 'motion':
                data_list.append(self.motion_RMS(path_list))
                data_list.append(self.motion_FD(path_list)[0])
                data_list.append(self.motion_FD(path_list)[1])
            elif self.other_type == 'brain_size':
                data_list.append(self.get_brainsize(path_list))

            data_list=np.array(data_list).T
                
            out_file_name = output_path+file_type+'/'+output+'.csv'

        pd_data_list = pd.DataFrame(data_list,index=self.subid)
        pd_data_list.to_csv(out_file_name)
        # with open(out_file_name,'w',newline = '') as f:
        #         f_csv = csv.writer(f)path
        #         f_csv.writerows(data_list)1
if __name__=='__main__':
    labelpath = 'E:\projects\genetic_imaging\HCPdata\VanEssenMap\HCP_PhaseTwo\Q1-Q6_RelatedParcellation210\MNINonLinear\\fsaverage_LR32k\\Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii'

    labeldata =  iofiles.CIFTI(labelpath).read_cifti()
    getdata=get_hcp_data(HCP_TEST_DATA)
    # pathlist = getdata.get_file_path_list('other')
    # print(getdata.motion_FD(pathlist))
    # print(memory_usage(getdata.getsave_certain_data('func',labeldata,'testfortime',data_out_file)))
    getdata.getsave_certain_data('other', labeldata, 'testfortime', data_out_file)



