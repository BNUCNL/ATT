import numpy as np
from atlas import UserDefinedException


class stats(object):
    def __init__(self, analyzer):
        self.meas = np.copy(analyzer.meas)
        self.subj_gender = list(analyzer.subj_gender)
        self.subj_id = list(analyzer.subj_id)
        self.analyzer = analyzer

    def bootstrap(self, aoi, beh=None, n_repeat=100, feat_sel=None):
        """

        Parameters
        ----------
        aoi : analysis of interest
        n_repeat : number of repeat
        feat_sel: feat selector

        Returns
        -------

        """
        n_sample = self.meas.shape[0]
        if feat_sel is None:
            n_feat = self.meas.shape[1]
        else:
            n_feat = len(feat_sel)

        if beh is not None:
            if beh.ndim == 1:
                beh = np.expand_dims(beh, axis=1)

        boot_index = np.random.randint(n_sample, size=(n_sample, n_repeat))
        if aoi is 'behavior_predict1':
            boot_stats = np.empty((n_feat, beh.shape[1], n_repeat))
            boot_stats.fill(np.nan)
            print boot_stats.shape
            for i in np.arange(n_repeat):
                index = boot_index[:, i]
                self.analyzer.meas = self.meas[index, :]
                self.analyzer.subj_id = [self.subj_id[s] for s in index]
                self.analyzer.subj_gender = [self.subj_gender[s] for s in index]
                rs = self.analyzer.behavior_predict1(beh[index, :], ['fakeBeh'])
                boot_stats[:, :, i] = rs[0]
        else:
            raise UserDefinedException('Wrong analysis of interest(aoi)!')

        return boot_stats

    def permutation(self, aoi, beh=None,  n_repeat=100, feat_sel=None):
        """

        Parameters
        ----------
        aoi
        beh
        n_repeat
        feat_sel

        Returns
        -------

        """

        n_sample = self.meas.shape[0]
        if feat_sel is None:
            n_feat = self.meas.shape[1]
        else:
            n_feat = len(feat_sel)

        if aoi is 'behavior_predict1':
            perm_stats = np.empty((n_feat, beh.shape[1], n_repeat))
            perm_stats.fill(np.nan)
            for i in np.arange(n_repeat):
                perm_index = np.random.permutation(n_sample)
                perm_stats[:, :, i] = self.analyzer.behavior_predict1(beh[perm_index])
        else:
            raise UserDefinedException('Wrong analysis of interest(aoi)!')

        return perm_stats


    def split_half_reliability(self, aoi, beh=None,  n_repeat=100, feat_sel=None):
        """

        Parameters
        ----------
        beh
        n_repeat
        feat_sel

        Returns
        -------

        """

        n_sample = self.meas.shape[0]
        half = np.fix(n_sample/2).astype(int)
        if feat_sel is None:
            n_feat = self.meas.shape[1]
        else:
            n_feat = len(feat_sel)

        if aoi is 'behavior_predict1':
            sph_stats = np.empty((n_feat, beh.shape[1], n_repeat, 2))
            sph_stats.fill(np.nan)

            for i in np.arange(n_repeat):
                index = np.random.permutation(n_sample)
                fold_index = list()
                fold_index.append(index[np.arange(0, half)])
                fold_index.append(index[np.arange(half, n_sample)])
                for j in (0, 1):
                    self.analyzer.meas = self.meas[fold_index[j], :]
                    self.analyzer.subj_id = [self.subj_id[s] for s in fold_index[j]]
                    self.analyzer.subj_gender = [self.subj_gender[s] for s in fold_index[j]]
                    sph_stats[:, :, i, j] = self.analyzer.behavior_predict1(beh[fold_index[j]])
        else:
            raise UserDefinedException('Wrong analysis of interest(aoi)!')

        return sph_stats