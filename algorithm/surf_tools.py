# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np
from . import tools
import copy

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
        assert edge_node_num.count((edge_node_num[0]) == len(edge_node_num)), "One edge should only contain 2 nodes"
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

def caloverlap(imgdata1, imgdata2, label1, label2, index = 'dice', controlsize = False, actdata = None):
    """
    Compute overlap (dice coefficient or percentage) in surface format data
    
    Parameters:
    ----------
    imgdata1, imgdata2: two image data in surface format
    label1, label2: label in correspond imgdata that used for overlap
    index: output index, dice or percent
    controlsize: whether control roi size of label in imgdata2 to roi size of label in imgdata1 or not, the method is to extract region with highest activation and paired them with region size of imgdata1
    actdata: by default is None, if controlsize is True, it's better to give actdata in this function

    Return:
    ----------
    overlapidx: overlap values
    
    Example: 
    ---------
    >>> overlapidx = caloverlap(imgdata1, imgdata2, label1, label2)
    """
    if imgdata1.ndim == 3:
        imgdata1 = imgdata1[:,0,0]
    if imgdata2.ndim == 3:
        imgdata2 = imgdata2[:,0,0]
    if controlsize is True:
        imgsize1 = imgdata1[imgdata1==label1].shape[0]
        try:
            imgdata2 = tools.control_lbl_size(imgdata2, actdata, imgsize1)   
        except AttributeError:
            raise Exception('We haven''t make it clear whether allow actdata as None when size need to be controlled, please input actdata here')
    overlap = np.logical_and(imgdata1==label1, imgdata2==label2)
    try:
        if index == 'dice':
            overlapidx = 2.0*len(overlap[overlap==1])/(len(imgdata1[imgdata1==label1])+len(imgdata2[imgdata2==label2]))
        elif index == 'percent':
            overlapidx = 1.0*len(overlap[overlap==1])/(len(imgdata1[imgdata1==label1]))
        else:
            raise Exception('please input dice or percent as parameters')
    except ZeroDivisionError as e:
        overlapidx = np.nan
    return overlapidx

def get_n_ring_neighbour(faces, n, option='part'):
    """
    Get n ring neighbour from faces array

    Parameters:
    ---------
    faces: faces array
    n: ring number
    option: 'part' or 'all'
            'part' means get the n_th ring neighbour
            'all' means get the n ring neighbour

    Return:
    ---------
    ringlist: array of ring node of each vertex
              The format of output will like below
              [{i1,j1,k1,...}, {i2,j2,k2,...}, ...]
              each element correspond to a vertex label

    Example:
    ---------
    >>> ringlist = get_n_ring_neighbour(faces, n)
    """
    n_vtx = np.max(faces) + 1 
    # find l_ring neighbours' id of each vertex
    n_ring_neighbours = [set() for _ in range(n_vtx)]
    for face in faces:
        for v_id in face:
            n_ring_neighbours[v_id].update(set(face))
    # remove vertex itself from its neighbour set
    for v_id in range(n_vtx):
        n_ring_neighbours[v_id].remove(v_id)
    if n == 1:
        return n_ring_neighbours
    # find n_ring_neighbours
    one_ring_neighbours = copy.deepcopy(n_ring_neighbours)
    n_th_ring_neighbours = copy.deepcopy(n_ring_neighbours)
    # n>1, go for further neighbours
    for i in range(n-1):
        for neighbour_set in n_th_ring_neighbours:
            neighbour_set_tmp = neighbour_set.copy()
            for v_id in neighbour_set_tmp:
                neighbour_set.update(one_ring_neighbours[v_id])
        if i == 0:
            for v_id in range(n_vtx):
                n_th_ring_neighbours[v_id].remove(v_id)
        for v_id in range(n_vtx):
            # get the (i+2)th ring neighbours
            n_th_ring_neighbours[v_id] -= n_ring_neighbours[v_id]
            # get the (i+2) ring neighbours
            n_ring_neighbours[v_id] |= n_th_ring_neighbours[v_id]
        
    if option == 'part':
        return n_th_ring_neighbours
    elif option == 'all':
        return n_ring_neighbours
    else:
        raise Exception('bad option!')
       
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
    if labelnum is None:
        labelnum = int(np.max(labels))
    masksize = []
    for i in range(labelnum):
        masksize.append(len(mask[mask == i+1]))
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
        labelnum = int(np.max(labels))

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

def generate_mask_by_labelid(labelId,labeldata):
    """
    you can genereate  mask using a lable ID 
    
    labelId: it can be a int or a list of label 
    labeldata: a mask with all label id in it 
      
    Example:
        generate_mask_by_labelid([1,2],labeldata)
    """
    mask = np.zeros_like(labeldata)
    if type(labelId) == int:
        mask[labeldata == labelId] = 1
    elif type(labelId) == list:
        for i in labelId:
            mask[labeldata == i] = 1
    else:
        raise Exception('not correte data type of labelId, please input an int or a list of labelId')
        
    return mask

def generate_mask_by_vernum(vertex_num,data,correspodig_matrix=None):
    """
    generate a mask by vertex number 
    
    vertex_num: a list or an array contain your vertex number of your ROI
    data: you should provide a one-dimensional data matrix as blueprint. stucture of mask generated will be just like your input data
    corresponding_matrix: consider there may be some mismathc between index of data matrix and vertex num (due to delete of some vertex from data matix),
    a correspond_matrix should be provided, if there is no missing vertex in your data, please ingore this parameters 
    
    Example:
        generate_mask_by_vertex(vertex_num,data,corresponding_matrix)
    """
    mask = np.zeros_like(data)
    if correspodig_matrix is None:
        for i in vertex_num:
            mask[i] = 1 
    else:
        for i,e in enumerate(correspodig_matrix):
            if e in vertex_num:
                mask[i] = 1
                    
    return mask    

 
