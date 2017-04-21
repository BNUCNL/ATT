# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 et:

import numpy as np

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
        node_number = len(ring_list)
        adjmatrix = np.zeros((node_number, node_number))
        for i,e in enumerate(ring):
            for j in e:
                adjmatrix[i,j] = 1
        return adjmatrix

def caldice(imgdata1, imgdata2, label1, label2):
    """
    Compute dice coefficient in surface format data
    
    Parameters:
    ----------
    imgdata1, imgdata2: two image data in surface format
    label1, label2: label in correspond imgdata that used for dice coefficient

    Return:
    ----------
    dice: dice value
    
    Example: 
    ---------
    >>> dice = caldice(imgdata1, imgdata2, label1, label2)
    """
    if imgdata1.ndim == 3:
        imgdata1 = imgdata1[:,0,0]
    if imgdata2.ndim == 3:
        imgdata2 = imgdata2[:,0,0]
    overlap = np.logical_and(imgdata1==label1, imgdata2==label2)
    dice = 2.0*len(overlap[overlap==1])/(len(imgdata1[imgdata1==label1])+len(imgdata2[imgdata2==label2]))
    return dice

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
    import copy
    n_vtx = np.max(faces) + 1 
    # find l_ring neighbours' id of each vertex
    n_ring_neighbours = [set() for _ in range(n_vtx)]
    for face in faces:
        for v_id in face:
            n_ring_neighbours[v_id].update(set(face))
    # remove vertex itself from its neighbour set
    for v_id in range(n_vtx):
        n_ring_neighbours[v_id].remove(v_id)
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
       





