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


stem_path = 'e:\\projects\\genetic_imaging\\HCPdata\\HCP900'
output_stem_path = 'E:\\projects\\genetic_imaging\\testResults'
 

class get_hcp_data(object):
    """
    usage
    >>>get_data = get_hcp_data(data_path)
    >>>get_data.getsave_certain_data('func',label_data,output_stem_path,face-avg_t.csv)
    """
    def __init__(self,stem_path):
        self.stem_path = stem_path
        subid = os.listdir(stem_path) #get subjects' id according to to folder's name
        self.subid = subid

    def motion_FD(self,path_list):
        power_rela_fd_mean = []
        dt_rela_fd_mean = []

        def cal_relaFD(re1, re2):
            """
            calculate each run's relative FD
            """
            relative_fd = 0
            for i in range(0, 3):
                relative_fd += abs(re1[i] - re2[i])
            for i in range(3, 6):
                relative_fd += 50 * math.pi / 150 * (abs(re1[i] - re2[i]))

            return math.sqrt(relative_fd)

        def cal_relaFD_mean(ls):
            """
            calculate mean relative FD across two runs
            """
            ls_float = [list(map(lambda x: float(x), j)) for j in ls]
            rela_fd = []
            for i in range(1, len(ls_float)):
                rela_fd.append(cal_relaFD(ls_float[i], ls_float[i - 1]))
            indi_fd_mean = np.mean(np.array(rela_fd))
            return indi_fd_mean

        for i in path_list:# calculate FD by power's method
            try:
                with open(os.path.join(i,'tfMRI_WM_LR','Movement_Regressors.txt'), 'r') as f:
                    s = f.readlines()
                    regre_LR = np.array([j.rstrip().split() for j in s])
                with open(os.path.join(i,'tfMRI_WM_RL','Movement_Regressors.txt'), 'r') as f:
                    s = f.readlines()
                    regre_RL = np.array([j.rstrip().split() for j in s])
                indi_FD_mean = (cal_relaFD_mean(regre_LR) + cal_relaFD_mean(regre_RL)) / 2
                power_rela_fd_mean.append(indi_FD_mean)
            except IOError as e:
                print(e)
                power_rela_fd_mean.append([])

        for i in path_list: # calculate FD directly through dt of the regressor
            try:
                with open(os.path.join(i,'tfMRI_WM_LR','Movement_Regressors_dt.txt'), 'r') as f:
                    s = f.readlines()
                    regre_LR = np.array([j.rstrip().split() for j in s])
                with open(os.path.join(i,'tfMRI_WM_RL','Movement_Regressors_dt.txt'), 'r') as f:
                    s = f.readlines()
                    regre_RL = np.array([j.rstrip().split() for j in s])

                indi_FD_mean = (cal_relaFD_mean(regre_LR) + cal_relaFD_mean(regre_RL)) / 2
                dt_rela_fd_mean.append(indi_FD_mean)

            except IOError as e:
                print(e)
                dt_rela_fd_mean.append([])

        return power_rela_fd_mean,dt_rela_fd_mean

    def get_brainsize(self,path_list):
        """
        # read volume size in the file aseg.stats
        
        """
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
                with open(os.path.join(i,'tfMRI_WM_LR','Movement_RelativeRMS_mean.txt'), 'r') as f:
                    relative_LR = f.readline()
                with open(os.path.join(i,'tfMRI_WM_RL','Movement_RelativeRMS_mean.txt'), 'r') as f:
                    relative_RL = f.readline()
                relative_meanRMS.append((float(relative_LR) + float(relative_RL) / 2))
            except IOError as e:
                print(e)
                relative_meanRMS.append([])
        return relative_meanRMS

    @decorators.timer
    def get_path_list(self,file_type):
        """
        generate a list containing needed file path
        file_type: func
                   stru
                   other
        """
        if file_type == 'func':

            func_sample_path = os.path.join(self.stem_path, '100307', 'MNINonLinear', 'Results', 'tfMRI_WM')
            func_stem_path = os.path.join('MNINonLinear', 'Results', 'tfMRI_WM')

            catagory_num = int(input('''
    please input the number of  catagoty you want:
    1.*.hp200_s12_level12.feat 
    2.*.hp200_s2_level2.feat 
    3.*.hp200_s2_level2_MSMALL.feat
    4.*.hp200_s4_level2.feat
    5.*.hp200_s4_level2_MSMALL.feat
    6.**.hp200_s8_level2.feat
    --
            '''))
            datatype_num = int(input('please input the number of your data type:\n:1.t value 2.beta\n--'))
            cope_num = int(input('''
                        please input the contrast type
                        1.body-avg
                        2.face-avg
                        3.place-avg
                        4.tool-avg
                        '''))

            catagory = os.listdir(func_sample_path) #get different type of folders
            cope_folders = ['cope19.feat', 'cope20.feat', 'cope21.feat', 'cope22.feat']
            data_type =['tstat1.dtseries.nii','cope1.dtseries.nii']
            self.catagory = catagory[catagory_num-1]

            path_list = [os.path.join(self.stem_path,i, func_stem_path, catagory[catagory_num - 1], 'GrayordinatesStats',cope_folders[cope_num - 1],data_type[datatype_num-1]) for i in self.subid]

        elif file_type == 'stru':
            self.catagory = ''
            stru_type_num = int(input('''
    please input the structure file you need (THE number) 
    1.*.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii
    2.*.curvature_MSMAll.32k_fs_LR.dscalar.nii
    3.*.thickness_MSMAll.32k_fs_LR.dscalar.nii
    --
            '''))
            brain_region_num = int(input('''
    please input number of your target region
    1. left hemisphere
    2. right hemishpere
    3. whole brain
    --
             '''))

            brain_region = ['.L','.R','']
            stru_type = ['.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii',
                         '.curvature_MSMAll.32k_fs_LR.dscalar.nii',
                         '.thickness_MSMAll.32k_fs_LR.dscalar.nii']

            path_list  = [os.path.join(self.stem_path,i, 'MNINonLinear', 'fsaverage_LR32k',i+brain_region[brain_region_num-1]+stru_type[stru_type_num-1]) for i in self.subid]


        elif file_type == 'other':
            other_type = int(input('''
            please input the type of data you want
            1.Motion
            2.brain_size
            '''))
            if other_type == 1:
                self.other_type ='motion'
                path_list = [os.path.join(self.stem_path,i,'MNINonLinear','Results') for i in self.subid]
            elif other_type == 2:
                self.other_type = 'brain_size'
                path_list = [os.path.join(self.stem_path,i,'T1w',i,'stats','aseg.stats') for i in self.subid]
        else:
            raise Exception('please input the right file type: func, stru, other')
        return path_list

    @decorators.timer
    def getsave_certain_data(self,file_type,label_data,output_stem_path,output_filename):
        """
        
        :param file_type: the type of data you want to get. It can be 'func','stru',or 'other'
        :param label_data: matrix cantains label_data
        :param output_filename: the name of the csv file. eg. face-avg_t.csv
        :param output_stem_path: 
        :return: nothing returned, only a csv file is created
        """
        path_list = self.get_path_list(file_type)
        data_list = []
        if file_type == 'func' or file_type=='stru':

            output_path = os.path.join(output_stem_path,file_type,self.catagory)  # if the folder is not exist, create it
            isExists = os.path.exists(output_path)
            if not isExists:
                os.makedirs(output_path)

            avg_list = []
            for i in path_list:
                try:
                    data = iofiles._CIFTI(i).load()
                    avg_list.append([np.average(data[0:59412]),np.average(data[0:29696]),np.average(data[29696:59412])]) # get global signal for whole brain ,left brian, and right brain
                    data_list.append(surf_tools.get_signals(data, label_data))  # get roi mean value of each roi of each subject
                except IOError as e:
                    print(e)
                    avg_list.append([])
                    data_list.append([])
            pd_data_list = pd.DataFrame(data_list, index=self.subid, columns=list(range(1, 361)))
            avg_pd = pd.DataFrame(avg_list,index = self.subid,columns=['whole','left','right'])
            pd_data_list = pd.concat([pd_data_list,avg_pd],axis=1) #contact averaged roi signal and average global sigal


        elif file_type == 'other':
            output_path = os.path.join(output_stem_path,file_type)
            isExists = os.path.exists(output_path)
            if not isExists:
                os.makedirs(output_path)
            data_list = []

            if self.other_type == 'motion':
                data_list.append(self.motion_RMS(path_list))
                data_list.append(self.motion_FD(path_list)[0])
                data_list.append(self.motion_FD(path_list)[1])
            elif self.other_type == 'brain_size':
                data_list.append(self.get_brainsize(path_list))

            data_list=np.array(data_list).T
            pd_data_list = pd.DataFrame(data_list, index=self.subid)


        pd_data_list.to_csv(os.path.join(output_path,output_filename))



if __name__ == '__main__':
    import nibabel as nib
    label_img = nib.load('e:/coding/ATT/data/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii')
    label_data = label_img.get_data()[0]
    get_data = get_hcp_data(stem_path)
    get_data.getsave_certain_data('stru',label_data,output_stem_path,'curvature_msmall.csv')








