# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode:nil -*-
# vi: set ft=python sts=4 sw=4 et:

import numpy as np

def make_pm(mask, meth = 'all'):
    """
    Make probabilistic map
    ------------------------------
    Parameters:
        mask: mask
        meth: 'all' or 'part'. 
              all, all subjects are taken into account
              part, part subjects are taken into account
    Return:
        pm = probabilistic map
    """
    if mask.ndim != 4:
        raise Exception('Masks should be a 4D nifti file contains subjects')
    labels = np.unique(mask)[1:]
    pm = np.empty((mask.shape[0], mask.shape[1], mask.shape[2], labels.shape[0]))
    if meth == 'all':
        for i in range(labels.shape[0]):
            pm[..., i] = np.mean(mask == labels[i], axis = 3)
    elif meth == 'part':
        for i in range(labels.shape[0]):
            mask_i = mask == labels[i]
            subj = np.any(mask_i, axis = (0,1,2))
            pm[..., i] = np.mean(mask_i[..., subj], axis = 3)
    else:
        raise Exception('method not supported')
    return pm
        
def make_mpm(pm, threshold):
    """
    Make maximum probabilistic map (mpm)
    ---------------------------------------
    Parameters:
        threshold: threholds to mask probabilistic maps
    Return:
        mpm: maximum probabilisic map
    """
    pm_temp = np.empty((pm.shape[0], pm.shape[1], pm.shape[2], pm.shape[3]+1))
    pm_temp[..., range(1, pm.shape[3]+1)] = pm 
    pm_temp[pm_temp < threshold] = 0
    mpm = np.argmax(pm_temp, axis=3)
    return mpm    

def sphere_roi(voxloc, radius, value, datashape = [91,109,91]):
    """
    Generate a sphere roi which centered in (x,y,z)
    Parameters:
        data: your output data
        voxloc: (x,y,z), center vox of spheres
        radius: radius (unit: vox), note that it's a list
        value: label value 
    output:
        data: sphere roi image data
        loc: sphere roi coordinates
    """
    loc = []
    data = np.zeros(datashape)
    for n_x in range(int(voxloc[0]-radius[0]), int(voxloc[0]+radius[0]+1)):
        for n_y in range(int(voxloc[1]-radius[1]), int(voxloc[1]+radius[1]+1)):
            for n_z in range(int(voxloc[2]-radius[2]), int(voxloc[2]+radius[2]+1)):
                n_coord = np.array((n_x, n_y, n_z))
                coord = np.array((voxloc[0], voxloc[1], voxloc[2]))
                minus = coord - n_coord
                if (np.square(minus) / np.square(np.array(radius)).astype(np.float)).sum()<=1:
                    try:
                        data[n_x, n_y, n_z] = value
                        loc.append([n_x,n_y,n_z])
                    except IndexError:
                        pass
    loc = np.array(loc)
    return data, loc

def region_growing(image, coordinate, voxnumber):
    """
    Region growing
    Parameters:
        image: nifti data
        coordinate: raw coordinate
        voxnumber: max growing number
    Output:
        rg_image: growth region image
        loc: region growth location
    """
    loc = []
    nt = voxnumber
    tmp_image = np.zeros_like(image)
    rg_image = np.zeros_like(image)
    image_shape = image.shape
    
    x = coordinate[0]
    y = coordinate[1]
    z = coordinate[2]

    # ensuring the coordinate is in the image
    # inside = (x >= 0) and (x < image_shape[0]) and (y >= 0) and \
    #          (y <= image_shape[1]) and (z >= 0) and (z < image_shape[2])
    # if inside is not True:
    #     print "The coordinate is out of the image range"
    #     return False

    # initialize region_mean and region_size
    region_mean = image[x,y,z]
    region_size = 0
    
    # initialize neighbour_list with 10000 rows 4 columns
    neighbour_free = 10000
    neighbour_pos = -1
    neighbour_list = np.zeros((neighbour_free, 4))

    # 26 direct neighbour points
    neighbours = [[1,0,0],\
                  [-1,0,0],\
                  [0,1,0],\
                  [0,-1,0],\
                  [0,0,-1],\
                  [0,0,1],\
                  [1,1,0],\
                  [1,1,1],\
                  [1,1,-1],\
                  [0,1,1],\
                  [-1,1,1],\
                  [1,0,1],\
                  [1,-1,1],\
                  [-1,-1,0],\
                  [-1,-1,-1],\
                  [-1,-1,1],\
                  [0,-1,-1],\
                  [1,-1,-1],\
                  [-1,0,-1],\
                  [-1,1,-1],\
                  [0,1,-1],\
                  [0,-1,1],\
                  [1,0,-1],\
                  [1,-1,0],\
                  [-1,0,1],\
                  [-1,1,0]]
    while region_size < nt:
        # (xn, yn, zn) stored direct neighbour of seed point
        for i in range(6):
            xn = x + neighbours[i][0]
            yn = y + neighbours[i][1]
            zn = z + neighbours[i][2]
            
            inside = (xn >= 0) and (xn < image_shape[0]) and (yn >=0) and \
                 (yn < image_shape[1]) and (zn >= 0) and (zn < image_shape[2])
            # ensure the original flag 0 is not changed
            if inside and tmp_image[xn, yn, zn] == 0:
                neighbour_pos = neighbour_pos + 1
                neighbour_list[neighbour_pos] = [xn, yn, zn, image[xn,yn,zn]]
                tmp_image[xn,yn,zn] = 1
 
        # ensure there's enough space to store neighbour_list
        if (neighbour_pos + 100 > neighbour_free):
            neighbour_free += 10000
            new_list = np.zeros((10000,4))
            neighbour_list = np.vstack((neighbour_list, new_list))
        
        # the distance between every neighbour point value to new region mean value
        distance = np.abs(neighbour_list[:neighbour_pos+1, 3] - np.tile(region_mean, neighbour_pos+1))

        # chose min distance point
        index = distance.argmin()

        # mark the new region point with 2 and update new image
        tmp_image[x, y, z] = 2
        rg_image[x, y, z] = image[x, y, z]
        loc.append([x,y,z])
        region_size+=1
        
        # (x,y,z) the new seed point
        x = neighbour_list[index][0]
        y = neighbour_list[index][1]
        z = neighbour_list[index][2]
        
        # update region mean value
        region_mean = (region_mean*region_size + neighbour_list[index, 3])/(region_size + 1)
         
        # remove the seed point from neighbour_list
        neighbour_list[index] = neighbour_list[neighbour_pos]
        neighbour_pos -= 1

    loc = np.array(loc)
    return rg_image, loc

def peakn_location(data, ncluster = 5, rgsize = 10, reverse = False):
    """
    Using region growth to extract highest/lowest clusters
    --------------------------------------
    Parameters:
        data: raw data
        ncluster: cluster numbers by using information of data values
        rgsize: region growth size (voxel), constraint neighbouring voxels
        reverse: if True, get locations start from the largest values
                 if False, start from the lowest values
    Return:
        nth_loc: list of locations
    """
    if reverse is True:
        filterdata = np.argmin
    else:
        filterdata = np.argmax
    median_data = np.median(data)
    nth_loc = []
    for i in range(ncluster):
        temploc = np.unravel_index(filterdata(data), data.shape)
        nth_loc.append(temploc)
        tempdata, loc_rg = region_growing(data, temploc, rgsize)
        for j in loc_rg:
            data[j[0], j[1], j[2]] = median_data
    return nth_loc, tempdata

