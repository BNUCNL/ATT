# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
    
import math
import numpy as np
import os
from ATT.iofunc import iofiles
from ATT.algorithm import surf_tools
import pandas as pd
from ATT.util import decorators
from sklearn.decomposition.pca import PCA
import nibabel as nib
 

class hcpAWS(object):
    """
    usage
    >>>get_data = get_hcp_data(data_path)
    >>>get_data.getsave_certain_data('func',label_data,output_stem_path,face-avg_t.csv)
    """
    def __init__(self,stem_path):
        """
        Parameters:
            stem_path: the path where you put your hcp data in e.g, c:/hcp
        """
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

            #func_sample_path = os.path.join(self.stem_path, '100307', 'MNINonLinear', 'Results', 'tfMRI_WM')
            task_num = int(input('''
            please input the number of task:
                1.emotion
                2.languge
                3.gambling
                4.motor
                5.relational
                6.social
                7.working memory
            '''))
            task = ['tfMRI_EMOTION',
                    'tfMRI_GAMBLING',
                    'tfMRI_LANGUAGE',
                    'tfMRI_MOTOR',
                    'tfMRI_RELATIONAL',
                    'tfMRI_SOCIAL',
                    'tfMRI_WM']
            func_stem_path = os.path.join('MNINonLinear', 'Results', task[task_num-1])
            
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
                        please input the cope number: 
                        '''))
            
            catagory = ['tfMRI_WM_hp200_s12_level12.feat',
            'tfMRI_WM_hp200_s2_level2.feat',
            'tfMRI_WM_hp200_s2_level2_MSMAll.feat',
            'tfMRI_WM_hp200_s4_level2.feat',
            'tfMRI_WM_hp200_s4_level2_MSMAll.feat',
            'tfMRI_WM_hp200_s8_level2.feat']

            #catagory = os.listdir(func_sample_path) #get different type of folders
            cope_folder = 'cope'+str(cope_num)+'.feat'
            data_type =['tstat1.dtseries.nii','cope1.dtseries.nii']
            self.catagory = catagory[catagory_num-1]

            path_list = [os.path.join(self.stem_path,i, func_stem_path, catagory[catagory_num - 1], 'GrayordinatesStats',cope_folder,data_type[datatype_num-1]) for i in self.subid]
 
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
            
        print(path_list[0])
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
                pd_data_list = pd.DataFrame(data_list, index = ['RMS_mean','FD1','FD2'], columns=self.subid).T
            elif self.other_type == 'brain_size':
                data_list.append(self.get_brainsize(path_list))
                pd_data_list = pd.DataFrame(data_list,columns=self.subid).T

        pd_data_list.to_csv(os.path.join(output_path,output_filename))
        return pd_data_list
    
    @decorators.timer    
    def get_peak_value(self, pathlist, label_data, roi_id, output_file_name):
        all_sub_peak = []
        for i in pathlist:
            try:
                sub_img = nib.load(i)
                sub_data = sub_img.get_data()
                while sub_data.ndim > 1:
                    sub_data = sub_data[0]
                sub_max = [max(sub_data[0:len(label_data)][label_data == j]) for j in roi_id] #record each roi's peak value for each subject
                all_sub_peak.append(sub_max)
            except IOError as e:
                print(e)
                all_sub_peak.append([])
                     
        all_sub_peak = pd.DataFrame(all_sub_peak, index = self.subid,columns = roi_id)
        all_sub_peak.to_csv(output_file_name)
        return all_sub_peak
    
    @decorators.timer
    def get_PCA_component(self, pathlist, label_data, roi_id,output_filename,n_comp = [2,3,4,5]):
        all_sub_data = []
        for i in pathlist:
            try:
                sub_img = nib.load(i)
                sub_data = sub_img.get_data()
                while sub_data.ndim > 1:
                    sub_data = sub_data[0]
                all_sub_data.append(list(sub_data[0:len(label_data)]))
            except IOError as e:
                print(e)
                all_sub_data.append([])
        

        all_sub_data = pd.DataFrame(all_sub_data,index = self.subid).T # make this datafram has 59412 rows 
        all_sub_data = all_sub_data.dropna(axis = 1)
        for n in n_comp:
            pca = PCA(n_components = n)
            pca_component_ratio = []
            for j in roi_id:
                roi_data = all_sub_data.ix[label_data == j].T # get certain ROI's data
                pca.fit(roi_data)
                pca_component_ratio.append(pca.explained_variance_ratio_)
                
            pca_component_ratio = pd.DataFrame(pca_component_ratio,index = roi_id)
            pca_component_ratio.to_csv(output_filename[:-4] + str(n) + output_filename[-4:])
    
    @decorators.timer
    def get_all_vertex_data(self,pathlist,label_data,roi_id,output_filename):
        all_sub_data = []
        bool_matrix = (label_data == roi_id[0]) # merge False and Right matirx of ROIs in roi_id
        for j in roi_id[1:]:
            bool_matrix = bool_matrix | (label_data == j)
            
        try:
            for i in pathlist:
                try:
                    sub_img = nib.load(i)
                    sub_data = sub_img.get_data()
                    while sub_data.ndim > 1:
                        sub_data = sub_data[0]
                        
                    all_sub_data.append(list(sub_data[0:len(label_data)][bool_matrix]))
                except IOError as e:
                    print(e)
                    all_sub_data.append([])
                    
            all_sub_data = pd.DataFrame(all_sub_data,index = self.subid)
            all_sub_data.to_csv(output_filename)
        except MemoryError as e:
            print(e)
            print('we have load data of ', len(all_sub_data),' subjects')
            all_sub_data = pd.DataFrame(all_sub_data)
            all_sub_data.to_csv(output_filename)
        return bool_matrix
    
    @decorators.timer
    def get_all_roi_vertex_data(self,path_list,output_filename,ssid):
        all_sub_data = []
        pathlist = [path for path in path_list for sid in ssid if str(sid) in path]
        try:
            for i in pathlist:
                try:
                    sub_img = nib.load(i)
                    sub_data = sub_img.get_data()
                    while sub_data.ndim >1:
                        sub_data = sub_data[0]
                    all_sub_data.append(list(sub_data))
                except IOError as e:
                    print(e)
                    all_sub_data.append([])
            all_sub_data = pd.DataFrame(all_sub_data,index = ssid)
            all_sub_data.to_csv(output_filename)    
            
        except MemoryError as e:
            print(e)
            print('we have load data of ', len(all_sub_data),' subjects')
            all_sub_data = pd.DataFrame(all_sub_data)
            all_sub_data.to_csv(output_filename)
            
            
                
    
if __name__ == '__main__':
    stem_path = 'e:/projects/genetic_imaging/HCPdata/HCP900/'
    hcp = hcpAWS(stem_path)
    path = hcp.get_path_list('func')







