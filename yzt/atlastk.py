import os
import os.path
import numpy as np
from scipy import io as sio
from scipy import ndimage as nd
from scipy import stats
import scipy.spatial.distance as spdist
from pandas import DataFrame, Series
import nibabel as nib

from atlas import PairLabelSizeStats, AtlasStat, RoiStat, RoiProbMap, PeakKernelMap, AnatRelation, RankOrderMap, MultiRoiRelation, BetweenSubjRelation, dfkey
from datadeposite import DataDepositeTk

pjoin = os.path.join

class AtlasLabelingReliabilityTk(DataDepositeTk):
# Toolkit to calculate label reliability
# Include:1) inter reliablity
#	  2) intra reliablity
# 	  3) inter reliablity in each labeler
#	  4) intra reliability in each labeler
    def __init__(self, atlas_reliability):
        self.depo = atlas_reliability

    def cal_inter_label_size_stats(self, first_labels, second_labels):
# call function of _cal_pair_label_size_stats to return first label size,
# second label size,overlap label size and union label size into size
# and save it
        sizes = self._cal_pair_label_size_stats(first_labels, second_labels)
        self.depo.inter_label_size_stats = PairLabelSizeStats(*sizes)
    
    def cal_intra_label_size_stats(self, first_labels, second_labels):
# Do similiar operation as above
        sizes = self._cal_pair_label_size_stats(first_labels, second_labels)
        self.depo.intra_label_size_stats = PairLabelSizeStats(*sizes)
 
    def cal_inter_reliability(self, measure_name, measure_calculator):
# if inter_label_size_stats is not None
# execute _cal_reliability to calculate reliablity
        if self.depo.inter_label_size_stats is None:
            raise ValueError, 'Call Atlastk.cal_inter_label_size_stats first'
        self.depo.inter_reliability_measures[measure_name] = self._cal_reliability(
            measure_name, measure_calculator, self.depo.inter_label_size_stats)

    def cal_intra_reliability(self, measure_name, measure_calculator):
# Do similiar operation as above
        if self.depo.intra_label_size_stats is None:
            raise ValueError, 'Call Atlastk.cal_intra_label_size_stats first'
        self.depo.intra_reliability_measures[measure_name] = self._cal_reliability(
            measure_name, measure_calculator, self.depo.intra_label_size_stats)
    
    def cal_inter_reliability_each_labeler(self, measure_name, measure_calculator):
# very similiar with cal_inter_reliability,I guess it's used for reliability
# between different labeler
        if self.depo.inter_label_size_stats is None:
            raise ValueError, 'Call Atlastk.cal_inter_label_size_stats first'
        self.depo.inter_reliability_measures_each_labeler[measure_name] = self._cal_reliability_each_labeler(
            measure_name, measure_calculator, self.depo.inter_label_size_stats)

    def cal_intra_reliability_each_labeler(self, measure_name, measure_calculator):
# Do similiar operation as above
        if self.depo.intra_label_size_stats is None:
            raise ValueError, 'Call Atlastk.cal_intra_label_size_stats first'
        self.depo.intra_reliability_measures_each_labeler[measure_name] = self._cal_reliability_each_labeler(
            measure_name, measure_calculator, self.depo.intra_label_size_stats)

    def _cal_pair_label_size_stats(self, first_labels, second_labels):
# main function of cal_inter/intra_label_size_stats
# call label_size_stats for calculating first label size,second label size,
# overlap label size and union label size
        flabels = get_imgs_data(first_labels)
        slabels = get_imgs_data(second_labels)
        fsizes = DataFrame(index = ['atlas']+self.depo.labelnames,
                       columns = ['all_labelers']+self.depo.labelers)
        ssizes = DataFrame(index = ['atlas']+self.depo.labelnames,
                       columns = ['all_labelers']+self.depo.labelers)
        osizes = DataFrame(index = ['atlas']+self.depo.labelnames,
                       columns = ['all_labelers']+self.depo.labelers)
        usizes = DataFrame(index = ['atlas']+self.depo.labelnames,
                       columns = ['all_labelers']+self.depo.labelers)
        labelids = [None] + self.depo.labelids    # None for all labels
        for lbler in range(1+len(self.depo.labelers)):
            for roi in range(1+len(self.depo.labelnames)):
                dfix = (roi, lbler)
                (fsizes.ix[dfix], ssizes.ix[dfix], osizes.ix[dfix], 
                usizes.ix[dfix]) = label_size_stats(flabels[lbler], slabels[lbler], labelids[roi])
        return fsizes, ssizes, osizes, usizes    
    
    def _cal_reliability_each_labeler(self, mname, mcalculator, size_stats):
# 
        reliability = DataFrame(index = ['atlas',]+self.depo.labelnames,
                       columns = ['all_labelers',]+self.depo.labelers)
        for lbler in range(1+len(self.depo.labelers)):
            for roi in range(1+len(self.depo.labelnames)):
                dfix = (roi, lbler)
                reliability.ix[dfix] = mcalculator(
                    size_stats.first_label_sizes.ix[dfix], size_stats.second_label_sizes.ix[dfix],
                    size_stats.overlap_label_sizes.ix[dfix], size_stats.union_label_sizes.ix[dfix])
        return reliability
    
    def _cal_reliability(self, mname, mcalculator, size_stats):
        reliability = DataFrame(index = ['atlas',]+self.depo.labelnames,
                       columns = ['all_labelers',]+self.depo.labelers)
        for lbler in range(1+len(self.depo.labelers)):
            for roi in range(1+len(self.depo.labelnames)):
                dfix = (roi, lbler)
                reliability.ix[dfix] = mcalculator(
                    size_stats.first_label_sizes.ix[dfix], size_stats.second_label_sizes.ix[dfix],
                    size_stats.overlap_label_sizes.ix[dfix], size_stats.union_label_sizes.ix[dfix])
        return reliability


class AtlasStatTk(DataDepositeTk):
    def __init__(self, atlasstat):
        self.depo = atlasstat
        self.roitkset = map(RoiStatTk, atlasstat.roistatset)

    def size_thresh(self, sthr):
        roistatset = map(lambda roistattk: roistattk.size_thresh(sthr), self.roitkset)
        return AtlasStat(name=self.depo.name, roistatset=roistatset,
                         sthr=sthr, atlas=self.depo.atlas)

    def cal_point_corrdinates(self, signal):
        pk_center_tk = SignalTk([nd.maximum_position, nd.center_of_mass, lambda x, y: nd.center_of_mass(y, None)], [dfkey.peak, dfkey.gcenter, dfkey.center])
        self.extract_signals_stats([signal], pk_center_tk, [''])

    def extract_peak_vals(self, signals, sigkeys):
        self.extract_dfpoints_vals(dfkey.peak, signals, sigkeys)

    def extract_dfpoints_vals(self, ptkey, signals, sigkeys):
        for roitk in self.roitkset:
            roitk.extract_dfpoints_vals(ptkey, signals, sigkeys)

    def extract_signals_stats(self, signals, signaltk, sigkeys):
        for roitk in self.roitkset:
            roitk.extract_signals_stats(signals, signaltk, sigkeys)

    def update_cached_stats(self, func, inkeys, outkeys):
        for roitk in self.roitkset:
            roitk.update_cached_stats(func, inkeys, outkeys)

    def cal_count_map(self):
        for roitk in self.roitkset:
            roitk.cal_count_map()

    def cal_any_map(self):
        for roitk in self.roitkset:
            roitk.cal_any_map()

    def cal_prob_maps(self, all_sub=True):
        for roitk in self.roitkset:
            roitk.cal_prob_map(all_sub)

    def save_prob_maps(self, datadir, mapno=0):
        for roitk in self.roitkset:
            roitk.save_prob_map(datadir, self.depo.atlas.img_affine, mapno)

    def cal_anat_relations(self, relation_name, relation_calculator, anat_name, anat_img, *args):
        for roitk in self.roitkset:
            roitk.cal_anat_relations(relation_name, relation_calculator, anat_name, anat_img, *args)

    def peak_kernel_map(self, peakkey, sigma):
        for roitk in self.roitkset:
            roitk.peak_kernel_map(peakkey, sigma)

    def rank_order_map(self, signal, key, all_subj=True):
        for roitk in self.roitkset:
            roitk.rank_order_map(signal, key, all_subj)

    def cal_max_prob_map(self, pthr, mapno=0, save=False, filename=None):
        probmaplist = [roitk.depo.prob_maps[mapno].probmap
                       for roitk in self.roitkset]
        self.depo.max_prob_map = max_prob_roi(probmaplist, pthr)
        if save:
            save_nif(self.depo.max_prob_map, self.depo.atlas.img_affine, filename)

    def merge_roi_probs_nonoverlap():
        pass

    def cal_between_subj_relation(self, relation_name, relation_calculator, cal_reflexive=True, reflexive_relation=1):
        for roitk in self.roitkset:
            roitk.cal_between_subj_relation(relation_name, relation_calculator, cal_reflexive, reflexive_relation)

    def cal_multi_roi_relation(self, relation_name, relation_calculator, subject_data_extractor, cal_reflexive=True, reflexive_relation=0):
        nroi = len(self.depo.roistatset)
        nsubj = self.depo.atlas.nsubjs
        relation_matrix = np.zeros((nsubj, nroi, nroi))
        roitkset = self.roitkset
        for s in range(nsubj):
            for i in range(nroi):
                if cal_reflexive:
                    relation_matrix[s][i][i] = reflexive_relation
                else:
                    relation_matrix[s][i][i] = np.nan
                if not self.roitkset[i].has_subject(s):
                    relation_matrix[s][i] = np.nan
                else: 
                    for j in range(i+1, nroi):
                        if not self.roitkset[j].has_subject(s):
                            relation_matrix[s][i][j] = np.nan
                        else:
                            relation_matrix[s][i][j] = relation_calculator(
                                subject_data_extractor(roitkset[i], s), 
                                subject_data_extractor(roitkset[j], s), 
                                )
            relation_matrix[s] = reflect_complet_a_matrix(relation_matrix[s])
        mroi_relation = MultiRoiRelation(relation_matrix=relation_matrix, 
                        relation_name=relation_name, 
                        size_thresh=self.depo.sthr)
        self.depo.multi_roi_relations[relation_name] = mroi_relation

    def stat_data_transform(self, transform_func, *args, **kargs):
        transformed_data = [roitk.stat_data_transform(transform_func, *args, **kargs) 
                            for roitk in self.roitkset]
        return transformed_data


class RoiStatTk(DataDepositeTk):
    def __init__(self, roistat):
        self.depo = roistat

    def size_thresh(self, sthr=0, wraptk=False):
        df = self.depo.df
        df = df[df[dfkey.size]>sthr]
        rs = RoiStat(self.depo.name, sthr, df, self.depo.roi)
        if wraptk:
            return RoiStatTk(rs)
        else:
            return rs

    def extract_signals_stats(self, signals, signaltk, sigkeys):
        for signal, key in zip(signals, sigkeys):
            self.extract_signal_stats(signal, signaltk, key)

    def extract_signal_stats(self, signal, signaltk, sigkey):
        res = map(lambda i:signaltk.extract_signal(signal[...,i], self.depo.roi.label[...,i]),
                  self.depo.df[dfkey.idx])
        res = np.array(res)
        pres = signaltk.get_func_pres()
        for j, keyname in enumerate(map(lambda x: x+'_'+sigkey, pres)):
            if keyname.endswith('_'):
                keyname = keyname.rstrip('_')
            if res[:,j].shape > 1:  # if you want to add an array for each row of the DataFrame
                self.depo.df[keyname] = list(res[:,j])
            else:
                self.depo.df[keyname] = res[:,j]

    def extract_dfpoints_vals(self, ptkey, signals, sigkeys):
        points = self.depo.df[ptkey].tolist()
        keys = [ptkey + '_' + sigkey for sigkey in sigkeys]
        self.extract_points_vals(points, signals, keys)

    def extract_points_vals(self, points, signals, keys):
        array_idx = self._get_4darray_index(points)
        for sig, key in zip(signals, keys):
            self.depo.df[key] = sig[array_idx]

    def update_cached_stats(self, func, inkeys, outkeys):
        for ink, outk in zip(inkeys, outkeys):
            if isinstance(ink, str):
                ink = [ink]
            else:
                ink = list(ink)
            self.depo.df[outk] = func(*[self.depo.df[k] for k in ink])

    def cal_count_map(self):
        res = np.sum(self.depo.roi.label[...,self.depo.df[dfkey.idx]], axis=3)
        self.depo.count_map = res

    def cal_any_map(self):
        res = np.any(self.depo.roi.label[...,self.depo.df[dfkey.idx]], axis=3)
        self.depo.any_map = res

    def cal_prob_map(self, allsubj=True):
        if self.depo.count_map is None:
            self.cal_count_map()
        res = self.depo.count_map
        if allsubj:
            res = np.true_divide(res, self.depo.roi.nsubj)
        else:
            res = np.true_divide(res, self.depo.nsubj)
        self.depo.prob_maps.append(RoiProbMap(probmap=res, allsubj=allsubj))
    
    def save_prob_map(self, datadir, affine, mapno=0):
        map_name = pjoin(datadir, self.depo.name+'.nii.gz')
        save_nif(self.depo.prob_maps[mapno].probmap, affine, map_name)

    def peak_kernel_map(self, peakkey, sigma):
        shape = self.depo.roi.label.shape[:3]
        kernelmap = points_kernel_map(self.depo.df[peakkey], shape, sigma)
        self.depo.peak_kernel_maps.append(PeakKernelMap(kernelmap=kernelmap, 
                                                   sigma=sigma))

    def cal_anat_relations(self, relation_name, relation_calculator, anat_name, anat_img, *args):
        if self.depo.any_map is None:
            self.cal_any_map()
        anymap = self.depo.any_map
        anat = get_img_data(anat_img)
        anat_ids = np.unique(anat)
        nid = len(anat_ids)
        #n_nz_id = anat_ids[anat_ids != 0]
        overlap_ids = np.unique(anat[anymap].flatten())
        #overlap_ids = overlap_ids[overlap_ids !=0 ]
        #res = np.zeros((self.depo.roi.nsubj, n_nz_id))
        res = np.zeros((self.depo.roi.nsubj, nid))
        res[:] = np.nan
        for idx in self.depo.df[dfkey.idx]:
            res[idx] = relation_calculator(self.depo.roi.label[...,idx], 
                                           anat, overlap_ids, nid, *args)
        keyname = relation_name+'_'+anat_name
        anat_relation = AnatRelation(relation_matrix=res,
                                     relation_name=keyname)
        self.depo.anat_relations[keyname] = anat_relation
            
    def cal_between_subj_relation(self, relation_name, relation_calculator, cal_reflexive=True, reflexive_relation=1, *args):
        res = relation_calculator(self.depo.roi.label)
        between_subj_relation = BetweenSubjRelation(relation_matrix=res,
                                         relation_name=relation_name)
        self.depo.between_subj_relations[relation_name] = between_subj_relation
        
    def rank_order_map(self, signal, allsubj=True):
        rank_tk = SignalTk([cal_rank], ['rank'])
        res = map(lambda j, i: rank_tk.extract_signal(signal[...,i], self.depo.roi.label[...,i])[0]/float(self.depo.df.size[j]), 
                  enumerate(self.depo.df[dfkey.idx]))
        ranksum = reduce(np.add, res)
        if allsubj:
            rankmean = np.true_divide(ranksum , self.depo.roi.nsubj)
        else:
            rankmean = np.true_divide(ranksum , self.depo.nsubj)
        self.depo.rank_order_maps.append(RankOrderMap(rankmap=rankmean, allsubj=allsubj))

    def has_subject(self, subj_idx):
        flag = self.depo.subj_flag[subj_idx]
        return flag

    def get_subject_label(self, subj_idx):
        label = self.depo.roi.label[...,subj_idx]
        return label

    def get_subject_peak(self, subj_idx):
        subject_id = self.depo.roi.subjs[subj_idx]
        peaks = self.depo.df[dfkey.peak]
        peak = peaks.ix[subject_id]
        #peak = np.array(peak)
        return peak
    
    def stat_data_transform(self, transform_func, *args, **kargs):
        return transform_func(self.depo, *args, **kargs)

    def _get_4darray_index(self, points):
        if len(points) != self.depo.nsubj:
            raise ValueError, 'Number of points not equal DataFrame length'
        pidx = np.hstack((points,
                  np.array(self.depo.df[dfkey.idx]).reshape(self.depo.nsubj, 1)))
        return zip(*pidx)


class SignalTk(object):
    def __init__(self, funcs, func_pres=None):
        self.funcs = list(funcs)
        if func_pres is None:
            func_pres = [''] * len(funcs)
        self.func_pres = list(func_pres)

    def extract_signal(self, signal, label, *args):
        res = []
        for func in self.funcs:
            res.append(func(signal, label, *args))
        return res

    def add_func(self, func, func_pre=''):
        self.funcs.append(func)
        self.func_pres.append(func_pre)

    def get_func_pres(self):
        return self.func_pres
            

def roistat_tomat(roistat, localizer, conditions, conditions_keys, contrast, thr, fwhm, dof, subjs_id, subjs_sex, resolution, affine, save=False, save_dir=None):
    df = roistat.df
    odf = DataFrame(index=subjs_id)
    odf = odf.reset_index().merge(df.reset_index(), how='left', on='index').set_index('index')
    d = {}
    d['info'] = {}
    d['subj'] = {}
    d['geo'] = {}
    d['mag'] = {}
    d['rlat'] = {}

    d['info']['name'] = roistat.name
    d['info']['localizer'] = localizer
    d['info']['cond'] = np.array(conditions, dtype=np.object)
    d['info']['contrast'] = contrast
    d['info']['thr'] = thr
    d['info']['fwhm'] = fwhm
    d['info']['flag'] = np.array((~np.isnan(odf.size)).astype(np.bool))
    d['info']['dof'] = dof
    d['subj']['id'] = np.array(subjs_id, dtype=np.object)
    d['subj']['sex'] = subjs_sex
    d['geo']['size'] = (odf.size*resolution[0]*resolution[1]*resolution[2]).tolist()
    peak = odf['peak']
    center = odf['center']
    gcenter = odf['gcenter']
    null_mask = odf['size'].isnull()
    t = odf[null_mask]
    dummy_series = Series([[np.nan]*3]*len(t),index=t.index)
    peak[null_mask] = dummy_series
    center[null_mask] = dummy_series
    gcenter[null_mask] = dummy_series
    peak = np.array(peak.tolist())
    center = np.array(center.tolist())
    gcenter = np.array(gcenter.tolist())
    d['geo']['peak'] = apply_affine_group(peak, affine, np.float)
    d['geo']['center'] = apply_affine_group(center, affine, np.float)
    d['geo']['gcenter'] = apply_affine_group(gcenter, affine, np.float)

    d['geo']['subj_dice'] = roistat.between_subj_relations['dice'].relation_matrix

    zkeys = ['Z'+key for key in conditions_keys]
    tkeys = ['T'+key for key in conditions_keys]
    bkeys = ['B'+key for key in conditions_keys]
    ekeys = ['E'+key for key in conditions_keys]

    d['mag']['pt'] = np.vstack([getattr(odf, key) for key in ['peak_'+key for key in tkeys]]).T
    d['mag']['pz'] = np.vstack([getattr(odf, key) for key in ['peak_'+key for key in zkeys]]).T
    d['mag']['pb'] = np.vstack([getattr(odf, key) for key in ['peak_'+key for key in bkeys]]).T
    d['mag']['pe'] = np.vstack([getattr(odf, key) for key in ['peak_'+key for key in ekeys]]).T

    d['mag']['rt'] = np.vstack([getattr(odf, key) for key in ['roi_'+key for key in tkeys]]).T
    d['mag']['rz'] = np.vstack([getattr(odf, key) for key in ['roi_'+key for key in zkeys]]).T
    d['mag']['rb'] = np.vstack([getattr(odf, key) for key in ['roi_'+key for key in bkeys]]).T
    d['mag']['re'] = np.vstack([getattr(odf, key) for key in ['roi_'+key for key in ekeys]]).T

    d['mag']['stm'] = np.vstack([getattr(odf, key) for key in ['mean_'+key for key in tkeys]]).T
    d['mag']['szm'] = np.vstack([getattr(odf, key) for key in ['mean_'+key for key in zkeys]]).T
    d['mag']['sbm'] = np.vstack([getattr(odf, key) for key in ['mean_'+key for key in bkeys]]).T
    d['mag']['stsd'] = np.vstack([getattr(odf, key) for key in ['std_'+key for key in tkeys]]).T
    d['mag']['szsd'] = np.vstack([getattr(odf, key) for key in ['std_'+key for key in zkeys]]).T
    d['mag']['sbsd'] = np.vstack([getattr(odf, key) for key in ['std_'+key for key in bkeys]]).T

    d['rlat']['overlap'] = roistat.anat_relations['overlap_HOcort_thr0'].relation_matrix

    if save:
        if save_dir is None:
            save_dir = os.get_cwd()
        save_dir = os.path.join(save_dir, roistat.name)
        make_dir(save_dir)
        outfile = os.path.join(save_dir, 'roi.mat')
        sio.savemat(outfile, {'roi':d})
    return d

def output_mats(atlas_stattk, mat_dir, localizer, conditions, conditions_keys, contrast, thr, fwhm, dof):
    atlas_stat = atlas_stattk.depo
    atlas_deposite = atlas_stat.atlas
    allroi_mats = atlas_stattk.stat_data_transform(roistat_tomat, 
            localizer, 
            conditions, conditions_keys, contrast, 
            thr, fwhm, dof, 
            atlas_deposite.subjs, [int(sex=='m') for sex in atlas_deposite.sex],
            atlas_deposite.img_resolution,
            atlas_deposite.img_affine, True, mat_dir)

    savedict = dict()
    for r in allroi_mats:
        savedict[r['info']['name']] = r
    sio.savemat(mat_dir+'/mroi.mat', {'mroi':savedict})

    roi_relation_dict = {}
    for itemkey in atlas_stat.multi_roi_relations.keys():
        roi_relation_dict[itemkey] = atlas_stat.multi_roi_relations[itemkey].relation_matrix
    sio.savemat(mat_dir+'/roi_relation.mat', roi_relation_dict)

###################### Outdated but maybe useful ####################
def cal_idxrel_sym(f, s, cal_self=False, self_val=0, *args, **kargs):
    slen = len(s)
    df = {}
    for i in range(slen):
        if np.isscalar(s[i]) and np.isnan(s[i]):
            df[s.index[i]] = Series(np.nan, index=s.index)
            continue
        res = []
        if cal_self:
            res.append(f(s[i], s[i]))
        else:
            res.append(self_val)
        for j in range(i+1, slen):
            if np.isscalar(s[j]) and np.isnan(s[j]):
                res.append(np.nan)
            else:
                res.append(f(s[i], s[j], *args, **kargs))
        df[s.index[i]] = Series(res, index=s.index[i:])
    df = DataFrame(df)
    df = df.combine_first(df.T)
    return df

def roi_label_relation(roi, f, df, label, label_ids, label_name, *args):
    dflabel = roi.get_dflabel(df)
    roi_label = {}
    for lbid in label_ids:
        res = []
        for i in range(dflabel.shape[3]):
            res.append(f(dflabel[...,i], label==lbid, *args))
        roi_label[label_name+'_'+str(lbid)] = res
    return DataFrame(roi_label, index=df.df.index)

def label_size_stats(image1, image2, labelid=None):
    if labelid is not None:
        image1 = (image1==labelid)
        image2 = (image2==labelid)
    else:
        image1 = (image1 > 0)
        image2 = (image2 > 0)
    size1 = np.sum(image1, (0,1,2))
    size2 = np.sum(image2, (0,1,2))
    overlap = np.sum(image1 & image2, (0,1,2))
    union = np.sum(image1 | image2, (0,1,2))

    return size1, size2, overlap, union

def max_prob_roi(probs, thresh=0):
    probs.insert(0, np.zeros(probs[0].shape))
    probs = np.array(probs)
    probs[probs<=thresh] = 0
    max = np.argmax(probs, axis=0)
    return max

def reflect_complet_a_matrix(m):
    shape = np.shape(m)
    if shape[0] != shape[1]:
        raise ValueError, 'Must be square matrix'
    length = shape[0]
    for j in range(length):
        for i in  range(j+1, length):
            m[i][j] = m[j][i]
    return m

def points_kernel_map(peaks, data_shape, sigma=1):
    res = np.zeros(data_shape)
    for p in peaks:
        res[p] += 1
    res = nd.gaussian_filter(res, sigma)
    return res

def cal_rank(signal, label):
    rank = np.argsort(signal[label]) + 1
    ranka = np.zeros(label.shape)
    ranka[label] = rank
    return ranka

def apply_affine(pos, mat, tp = float):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    if mat.shape != (4, 4):
        raise ValueError('Affine matrix should be 4 * 4')
    _x = mat[0,0] * x + mat[0,1] * y + mat[0,2] * z + mat[0,3]
    _y = mat[1,0] * x + mat[1,1] * y + mat[1,2] * z + mat[1,3]
    _z = mat[2,0] * x + mat[2,1] * y + mat[2,2] * z + mat[2,3]
    return tp(_x),tp(_y),tp(_z)

def apply_affine_group(points, mat, tp):
    return map(lambda x: apply_affine(x, mat, tp), points)

def get_img_data(img):
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, str):
        return nib.load(img).get_data()
    else:
        raise ValueError, 'Unknown Img Type: not a string and not a array'

def get_imgs_data(imgs):
    return map(get_img_data, imgs)

def save_nif(data, affine, name):
    if data.dtype.type is np.bool_:  # nibabel couldn't handle bool type
        data = data.astype(np.uint8)
    elif data.dtype.type is np.int64:
        data = data.astype(np.int32)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, name)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        raise ValueError, "%s has already exists and not a directory!" % path

def t_ztransform(t, dof):
    p = stats.t.cdf(t, dof)
    z = stats.norm.ppf(p)
    return z

def _cal_multi_ids(f):
    """Decorator for multi anatid operation.

    Parameters
    ----------
    f : func
        The function operates on a label and a anatid. f should accept a label image and a mask image with orther optional parameters.

    Returns
    -------
    multi_anatids_func : func
        The function could deal with a label and multiple anatids.

    Note
    ----
    Overlap with anatlabel 0 are put into the last element of the array.
    """
    def cal_multi(label, anat, anatids, nid, *args):
        res = np.array([np.nan] * nid)
        for anatid in anatids:
            res[anatid-1] = f(label, anat==anatid, *args)
        return res
    return cal_multi

def jaccard(maska, maskb):
    return float((maska & maskb).sum()) / (maska | maskb).sum()

def dice(maska, maskb):
    return (2. * (maska & maskb).sum()) / (maska.sum() + maskb.sum())

def hausdorff(maska, maskb, resolution=None):
    a = np.array(maska.nonzero()).T
    b = np.array(maskb.nonzero()).T
    if resolution is None or ((resolution[0] == resolution[1])and(resolution[1]==resolution[2])):
        mdist = spdist.cdist(a, b)
        if resolution is not None:
            mdist = mdist * resolution[0]
    else:
        mdist = spdist.cdist(a, b, lambda x, y: dist(x, y, resolution))
    ha = np.max(np.min(mdist, 1))
    hb = np.max(np.min(mdist, 0))
    return max(ha, hb)

def left_overlap_ratio(maska, maskb):
    return float((maska & maskb).sum()) / maska.sum()

multi_jaccard = _cal_multi_ids(jaccard)
multi_dice = _cal_multi_ids(dice)
multi_hausdorff = _cal_multi_ids(hausdorff)
multi_left_overlap = _cal_multi_ids(left_overlap_ratio)

def reflect_complet_a_matrix(m):
    """Complete a matrix by reflecting over the diagonal line."""
    shape = np.shape(m)
    if shape[0] != shape[1]:
        raise ValueError, 'Must be square matrix'
    length = shape[0]
    for j in range(length):
        for i in  range(j+1, length):
            m[i][j] = m[j][i]
    return m

def _cal_between_subjs(f):
    """Decorator for multi anatid operation.

    Parameters
    ----------
    f : func
        The function operates on a list of labels.

    Returns
    -------
    between_subject_func : func
        The function could deal with a list of labels.
    """
    def cal_between(labels, cal_reflexive=True, reflexive_relation=1, *args):
        length = np.shape(labels)[-1]
        res = np.zeros((length, length))
        for i in range(length):
            if cal_reflexive:
                res[i][i] = reflexive_relation
            else:
                res[i][i] = np.nan
            if np.sum(labels[...,i]) == 0:
                res[i] = np.nan
            else:
                for j in range(i+1, length):
                    if np.sum(labels[...,j]) == 0:
                        res[i][j] = np.nan
                    else:
                        res[i][j] = f(labels[...,i], labels[...,j])
        refleted_res = reflect_complet_a_matrix(res)
        return refleted_res
    return cal_between

between_jaccard = _cal_between_subjs(jaccard)
between_dice = _cal_between_subjs(dice)

def dist(v1, v2, res):
    return ((res[0] * (v1[0] - v2[0])) ** 2 +
            (res[1] * (v1[1] - v2[1])) ** 2 +
            (res[2] * (v1[2] - v2[2])) ** 2) ** 0.5

