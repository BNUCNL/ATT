# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
from . import tools
import copy
import random

def extract_edge_from_faces(faces):
    """
    Transfer faces relationship into edge relationship
    
    Parameters:
    ----------
    faces: faces array
    
    Return:
    -------
    edge: edge, format as [(i1,j1), (i2,j2), ...]

    Example:
    -------
    >>> edge = extract_edge_from_faces(faces)
    """
    import itertools
    edge = []
    for i,fa in enumerate(faces):
        c_gen = itertools.combinations(fa, 2)
        for j in c_gen:
            if j not in edge:
                edge.append(j)
        print('{} finished'.format(i))
    return edge

class GenAdjacentMatrix(object):
    """
    Generate adjacency matrix from edge or ring_list
    
    Return:
    --------
    ad_matrix: adjacent matrix
    
    Example:
    --------
    >>> gamcls = GenAdjacentMatrix()
    >>> admatrix = gamcls.from_edge(edge)
    """
    def __init__(self):
        pass

    def from_edge(self, edge):
        """
        Generate adjacent matrix from edge
        
        Parameters:
        -----------
        edge: edge list, which have the format like below, 
              [(i1,j1), (i2,j2), ...] 
              note that i,j is the number of vertex/node
        
        Return:
        -----------
        adjmatrix: adjacent matrix
        """ 
        assert isinstance(edge, list), "edge should be a list"
        edge_node_num = [len(i) for i in edge]
        assert edge_node_num.count(edge_node_num[0]) == len(edge_node_num), "One edge should only contain 2 nodes"
        node_number = np.max(edge)+1
        ad_matrix = np.zeros((node_number, node_number))
        for eg in edge:
            ad_matrix[eg] = 1
        ad_matrix = np.logical_or(ad_matrix, ad_matrix.T)
        ad_matrix = ad_matrix.astype('int')
        return ad_matrix

    def from_ring(self, ring):
        """
        Generate adjacent matrix from ringlist
        
        Parameters:
        ----------
        ring: list of ring node, the format of ring list like below
              [{i1,j1,k1,...}, {i2,j2,k2,...}, ...]
              each element correspond to a index (index means a vertex)
        
        Return:
        ----------
        adjmatrix: adjacent matrix 
        """
        assert isinstance(ring, list), "ring should be a list"
        node_number = len(ring)
        adjmatrix = np.zeros((node_number, node_number))
        for i,e in enumerate(ring):
            for j in e:
                adjmatrix[i,j] = 1
        return adjmatrix

def get_masksize(mask, labelnum = None):
    """
    Compute mask size in surface space
    
    Parameters:
    ----------
    mask: label image (mask)
    labelnum: mask's label number, use for group analysis

    Return:
    --------
    masksize: mask size of each roi

    Example:
    --------
    >>> masksize = get_masksize(mask)
    """
    if mask.ndim == 3:
        mask = mask[:,0,0]
    labels = np.unique(mask)[1:]
    masksize = []
    if len(labels) != 0:
        if labelnum is None:
            labelnum = int(np.max(labels))
        for i in range(labelnum):
            masksize.append(len(mask[mask == i+1]))
    else:
        masksize.append(0)
    return np.array(masksize)
    
def get_signals(atlas, mask, method = 'mean', labelnum = None):
    """
    Extract roi signals of atlas from mask
    
    Parameters:
    -----------
    atlas: atlas
    mask: mask, a label image
    method: 'mean', 'std', 'ste', 'max', 'vertex', etc.
    labelnum: mask's label numbers, add this parameters for group analysis

    Return:
    -------
    signals: signals of specific roi
   
    Example:
    -------
    >>> signals = get_signals(atlas, mask, 'mean')
    """
    if atlas.ndim == 3:
        atlas = atlas[:,0,0]
    if mask.ndim == 3:
        mask = mask[:,0,0]
    
    
    labels = np.unique(mask)[1:]
    if labelnum is None:
        try:
            labelnum = int(np.max(labels))
        except ValueError as e:
            print('value in mask are all zeros')
            labelnum = 0
    if method == 'mean':
        calfunc = np.nanmean
    elif method == 'std':
        calfunc = np.nanstd
    elif method == 'max':
        calfunc = np.max
    elif method == 'vertex':
        calfunc = np.array
    elif method == 'ste':
        calfunc = tools.ste
    else:
        raise Exception('Miss paramter of method')
    signals = []
    for i in range(labelnum):
        if np.any(mask==i+1):
            signals.append(atlas[mask==i+1])
        else:
            signals.append(np.array([np.nan]))
    return [calfunc(sg) for sg in signals]

def get_vexnumber(atlas, mask, method = 'peak', labelnum = None):
    """
    Get vertex number of rois from surface space data
    
    Parameters:
    -----------
    atlas: atlas
    mask: mask, a label image
    method: 'peak' ,'center', or 'vertex', 
            'peak' means peak vertex number with maximum signals from specific roi
            'vertex' means extract all vertex of each roi
    labelnum: mask's label numbers, add this parameters for group analysis
    
    Return:
    -------
    vexnumber: vertex number

    Example:
    --------
    >>> vexnumber = get_vexnumber(atlas, mask, 'peak')
    """
    if atlas.ndim == 3:
        atlas = atlas[:,0,0]
    if mask.ndim == 3:
        mask = mask[:,0,0]
    labels = np.unique(mask)[1:]
    if labelnum is None:
        try:
            labelnum = int(np.max(labels))
        except ValueError as e:
            labelnum = 0

    extractpeak = lambda x: np.unravel_index(x.argmax(), x.shape)[0]
    extractcenter = lambda x: np.mean(np.transpose(np.nonzero(x)))
    extractvertex = lambda x: x[x!=0]
    
    if method == 'peak':
        calfunc = extractpeak
    elif method == 'center':
        calfunc = extractcenter
    elif method == 'vertex':
        calfunc = extractvertex
    else:
        raise Exception('Miss parameter of method')

    vexnumber = []
    for i in range(labelnum):
        roisignal = atlas*(mask==(i+1))
        if np.any(roisignal):
            vexnumber.append(calfunc(roisignal))
        else:
            vexnumber.append(np.array([np.nan]))
    return vexnumber

def surf_dist(vtx_src, vtx_dst, one_ring_neighbour):
    """
    Distance between vtx_src and vtx_dst
    Measured by edge number
    
    Parameters:
    -----------
    vtx_src: source vertex, int number
    vtx_dst: destinated vertex, int number
    one_ring_neighbour: one ring neighbour matrix, computed from get_n_ring_neighbour with n=1
    the format of this matrix:
    [{i1,j1,...}, {i2,j2,k2}]
    each element correspond to a vertex label

    Return:
    -------
    dist: distance between vtx_src and vtx_dst

    Example:
    --------
    >>> dist = surf_dist(vtx_src, vtx_dst, one_ring_neighbour)
    """
    if len(one_ring_neighbour[vtx_dst]) == 1:
        return np.inf
    
    noderep = copy.deepcopy(one_ring_neighbour[vtx_src])
    dist = 1
    while vtx_dst not in noderep:
        temprep = set()
        for ndlast in noderep:
            temprep.update(one_ring_neighbour[ndlast])
        noderep.update(temprep)
        dist += 1
    return dist
  
def hausdoff_distance(imgdata1, imgdata2, label1, label2, one_ring_neighbour):
    """
    Compute hausdoff distance between imgdata1 and imgdata2
    h(A,B) = max{max(i->A)min(j->B)d(i,j), max(j->B)min(i->A)d(i,j)}
    
    Parameters:
    -----------
    imgdata1: surface image data1
    imgdata2: surface image data2
    label1: label of image data1
    label2: label of image data2
    one_ring_neighbour: one ring neighbour matrix, similar description of surf_dist, got from get_n_ring_neighbour

    Return:
    -------
    hd: hausdorff distance
    
    Example:
    --------
    >>> hd = hausdoff_distance(imgdata1, imgdata2, 1, 1, one_ring_neighbour)
    """
    imgdata1 = tools.get_specificroi(imgdata1, label1)
    imgdata2 = tools.get_specificroi(imgdata2, label2)
    hd1 = _hausdoff_ab(imgdata1, imgdata2, one_ring_neighbour) 
    hd2 = _hausdoff_ab(imgdata2, imgdata1, one_ring_neighbour)
    return max(hd1, hd2)
 
def _hausdoff_ab(a, b, one_ring_neighbour):
    """
    Compute hausdoff distance of h(a,b)
    part unit of function hausdoff_distance
    
    Parameters:
    -----------
    a: array with 1 label
    b: array with 1 label
    one_ring_neighbour: one ring neighbour matrix

    Return:
    -------
    h: hausdoff(a,b)

    """
    a = np.array(a)
    b = np.array(b)
    h = 0
    for i in np.flatnonzero(a):
        hd = np.inf
        for j in np.flatnonzero(b):
            d = surf_dist(i,j, one_ring_neighbour)    
            if d<hd:
                hd = copy.deepcopy(d)
        if hd>h:
            h = hd
    return h

def median_minimal_distance(imgdata1, imgdata2, label1, label2, one_ring_neighbour):
    """
    Compute median minimal distance between two images
    mmd = median{min(i->A)d(i,j), min(j->B)d(i,j)}
    for detail please read paper:
    Groupwise whole-brain parcellation from resting-state fMRI data for network node identification
    
    Parameters:
    -----------
    imgdata1, imgdata2: surface data 1, 2
    label1, label2: label of surface data 1 and 2 used to comparison
    one_ring_neighbour: one ring neighbour matrix, similar description of surf_dist, got from get_n_ring_neighbour
    
    Return:
    -------
    mmd: median minimal distance

    Example:
    --------
    >>> mmd = median_minimal_distance(imgdata1, imgdata2, label1, label2, one_ring_neighbour)
    """
    imgdata1 = tools.get_specificroi(imgdata1, label1)
    imgdata2 = tools.get_specificroi(imgdata2, label2)
    dist1 = _mmd_ab(imgdata1, imgdata2, one_ring_neighbour)
    dist2 = _mmd_ab(imgdata2, imgdata1, one_ring_neighbour)
    return np.median(dist1 + dist2)

def _mmd_ab(a, b, one_ring_neighbour):
    """
    Compute median minimal distance between a,b
    
    part computational completion of median_minimal_distance

    Parameters:
    -----------
    a, b: array with 1 label
    one_ring_neighbour: one ring neighbour matrix

    Return:
    -------
    h: minimal distance
    """
    a = np.array(a)
    b = np.array(b)
    h = []
    for i in np.flatnonzero(a):
        hd = np.inf
        for j in np.flatnonzero(b):
            d = surf_dist(i, j, one_ring_neighbour)
            if d<hd:
                hd = d
        h.append(hd)
    return h

def get_n_ring_neighbor(vertx, faces, n=1, ordinal=False):
    """
    Get n ring neighbour from faces array

    Parameters:
    ---------
    vertex: vertex number
    faces : the array of shape [n_triangles, 3]
    n : integer
        specify which ring should be got
    ordinal : bool
        True: get the n_th ring neighbor
        False: get the n ring neighbor

    Return:
    ---------
    ringlist: array of ring nodes of each vertex
              The format of output will like below
              [{i1,j1,k1,...}, {i2,j2,k2,...}, ...]
			  each index of the list represents a vertex number
              each element is a set which includes neighbors of corresponding vertex

    Example:
    ---------
    >>> ringlist = get_n_ring_neighbour(24, faces, n)
    """
    if isinstance(vertx, int):
        vertx = [vertx]
    nth_ring = [set([vx]) for vx in vertx]
    nring = [set([vx]) for vx in vertx]

    while n != 0:
        n = n - 1
        for idx, neighbor_set in enumerate(nth_ring):
            neighbor_set_tmp = [_get_connvex_neigh(vx, faces) for vx in neighbor_set]
            neighbor_set_tmp = set([item for sublist in neighbor_set_tmp for item in sublist])
            neighbor_set_tmp.difference_update(nring[idx])
            nth_ring[idx] = neighbor_set_tmp
            nring[idx].update(nth_ring[idx])
    if ordinal is True:
        return nth_ring
    else:
        return nring

def get_connvex(seedvex, faces, mask = None, masklabel = 1):
    """
    Get connected vertices that contain in mask
    We firstly need a start point to acquire connected vertices, then do region growing until all vertices in mask were included
    That means, output should satisfied two condition:
    1 overlap with mask
    2 connected with each other
    
    Parameters:
    -----------
    seedvex: seed point (start point)
    faces: faces array, vertex relationship
    mask: overlapping mask
    masklabel: specific mask label used as restriction

    Return:
    -------
    connvex: connected vertice set

    Example:
    --------
    >>> connvex = get_connvex(24, faces, mask)
    """
    connvex = set()
    connvex.add(seedvex)
    connvex_temp = _get_connvex_neigh(seedvex, faces, mask, masklabel)
    while not connvex_temp.issubset(connvex):
        connvex_dif = connvex_temp.difference(connvex)
        connvex.update(connvex_dif)
        connvex_temp = set()
        for sx in connvex_dif: 
            connvex_temp.update(_get_connvex_neigh(sx, faces, mask, masklabel))
        print('Size of sulcus {0}'.format(len(connvex)))
    return connvex  

def _get_connvex_neigh(seedvex, faces, mask = None, masklabel = 1):
    """
    Function to get neighbouring vertices of a seed that satisfied overlap with mask
    """
    if mask is not None:
        assert mask.shape[0] == np.max(faces) + 1 ,"mask need to have same vertex number with faces connection relationship"
    assert isinstance(seedvex, (int, np.integer)), "only allow input an integer as seedvex"

    raw_faces, _ = np.where(faces == seedvex)
    rawconnvex = np.unique(faces[raw_faces])
    if mask is not None:
        connvex = set()
        list_connvex = [i for i in rawconnvex if mask[i] == masklabel]
        connvex.update(list_connvex)
    else:
        connvex = set(rawconnvex)
    connvex.discard(seedvex)
    return connvex

def inflated_roi_by_rg(orig_mask, ref_mask, faces):
    """
    Inflate orig_mask by extracting connected parcel of ref_mask
    A detailed explanation of this method:
    We have a more stricted orig_mask which pointed relatively correct position of a specific region (e.g. hMT/V5+)
    To use a larger ref_mask as constraints, by extracting connected parcels in ref_mask (the seed point comes from orig_mask), we can get a larger relatively accurate ROI of a specific region

    Parameters:
    -----------
    orig_mask: original mask with a/several small parcel(s)
    ref_mask: reference mask with larger parcel(s)
    faces: relationship of connection
    """
    orig_maskset = set(np.where(orig_mask!=0)[0])
    ref_maskset = set(np.where(orig_mask!=0)[0])

    connvex = set()
    dif_connvex = orig_maskset.difference(connvex)
    parcel_num = 0
    while len(dif_connvex) != 0:
        seedpt = random.choice(tuple(dif_connvex))
        parcel_connvex = get_connvex(seedpt, faces, ref_mask)
        connvex.update(parcel_connvex)
        dif_connvex = orig_maskset.difference(connvex)
        parcel_num += 1
        print('parcel number: {0}'.format(parcel_num))
    return connvex

def cutrg2parcels(orig_mask, faces, label = 1):
    """
    Use region growing method to cut discontinuous specific regions to parcels

    Parameters:
    -----------
    orig_mask: original mask has a/several small parcel(s) with labels
    faces: relationship of connection
    label[int]: specific label of orig_mask that used to cut into parcels

    Returns:
    --------
    parcel_mask: parcel mask that generated from specific label of orig_mask

    Example:
    --------
    >>> parcel_mask = cutrg2parcels(orig_mask, faces, 1)
    """
    if not isinstance(label, int):
        raise Exception('label need to be an int')
    lbl_orig_mask = tools.get_specificroi(orig_mask, label)
    parcel_mask = np.zeros_like(orig_mask)

    orig_maskset = set(np.where(lbl_orig_mask!=0)[0])
    connvex = set()
    dif_connvex = orig_maskset.difference(connvex)

    parcel_num = 0
    while len(dif_connvex) != 0:
        seedpt = random.choice(tuple(dif_connvex))
        parcel_connvex = get_connvex(seedpt, faces, lbl_orig_mask, masklabel = label)
        connvex.update(parcel_connvex)
        dif_connvex = orig_maskset.difference(connvex)
        parcel_num += 1
        parcel_mask = tools.make_lblmask_by_loc(parcel_mask, tuple(parcel_connvex), parcel_num)
        print('parcel number: {0}'.format(parcel_num))
    return parcel_mask








