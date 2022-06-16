#!/usr/bin/env python3

import os
import sys
import time
import streamlit as st

import cv2
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import feature, filters, measure
from skimage.morphology import skeletonize

from plantcv import plantcv as pcv

from matplotlib.colors import LightSource
from scipy import interpolate
import scipy.linalg
import pandas as pd

import numpy as np
import copy
from scipy import ndimage

import shapefile
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiLineString
from shapely.ops import linemerge
from shapely.affinity import translate

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import geometry, ops
from matplotlib.path import Path
import numpy.ma as ma

import whitebox
import netCDF4
    
wbt = whitebox.WhiteboxTools()

@st.cache
def surfature(X,Y,Z):
    
    cellsize = np.abs(X[1, 2] - X[1, 1])
    
    gx, gy = np.gradient(Z,cellsize,cellsize)
    gxx, gxy1 = np.gradient(gx,cellsize,cellsize)
    gxy2, gyy = np.gradient(gy,cellsize,cellsize)
    
    gxy = 0.5*(gxy1+gxy2)
    gyx = gxy
    
    p = gx
    q = gy

    r = gxx
    t = gyy
    s = gxy

    H = - ( (1.0+q**2)*r - 2.0*p*q*s + ( 1.0 + p**2) * t ) / ( 2.0 * np.sqrt( (1.0+p**2+q**2)**3 ) )

    K = ( r*t - s**2 ) / ( 1.0 + p**2 + q**2 )**2

    M = np.sqrt( H**2 - K )

    #% Principle Curvatures
    Pmax = H + M
    Pmin = H - M

    n = 5
    m = 1
    
    # Pmin = np.sign(Pmin)*np.log(1.0+10**(n*m)*np.abs(Pmin))
    # Pmax = np.sign(Pmax)*np.log(1.0+10**(n*m)*np.abs(Pmax))

    return Pmax,Pmin

@st.cache
def read_asc(ascii_file):

    import numpy as np
    from linecache import getline

    source1 = ascii_file
    # Parse the header using a loop and
    # the built-in linecache module
    hdr = [getline(source1, i) for i in range(1, 7)]
    values = [float(h.strip('\n').strip().split(" ")[-1]) for h in hdr]
    cols, rows, xll, yll, cell, nd = values
    cols = int(cols)
    rows = int(rows)
    delta_x = cell
    delta_y = cell

    values = [(h.split(" ")[0]) for h in hdr]
    s1, s2, s3, s4, s5, s6 = values

    if (s3 == 'xllcorner'):

        x_min = xll + 0.5 * delta_x
        y_min = yll + 0.5 * delta_y

    elif (s3 == 'xllcenter'):

        x_min = xll
        y_min = yll

    x_max = x_min + (cols - 1) * delta_x
    x = np.linspace(x_min, x_max, cols)

    y_max = y_min + (rows - 1) * delta_y
    y = np.linspace(y_min, y_max, rows)

    X, Y = np.meshgrid(x, y)

    # Load the dem into a numpy array
    h = np.flipud(np.loadtxt(source1, skiprows=6))

    h[h == nd] = np.nan

    return (X, Y, h, x_min, x_max, delta_x, y_min, y_max, delta_y)


@st.cache
def reverse_geom(geom):
    def _reverse(x, y, z=None):
        if z:
            return x[::-1], y[::-1], z[::-1]
        return x[::-1], y[::-1]

    return ops.transform(_reverse, geom)


@st.cache
def offset_path(medial_axis, offset, plot_flag):

    x, y = medial_axis.coords.xy
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    print('x_Avg,y_avg', x_avg, y_avg)

    # medial_axis = translate(medial_axis, xoff=-x_avg, yoff=-y_avg)

    offset_l = medial_axis.parallel_offset(offset, 'left', join_style=1)
    offset_r = medial_axis.parallel_offset(offset, 'right', join_style=1)

    # print(offset_l.geom_type)
    # print(offset_l)
    # print(offset_r.geom_type)
    # print(offset_r)

    if offset_l.geom_type == 'MultiLineString':

        offset = offset_r

    elif offset_r.geom_type == 'MultiLineString':

        offset = offset_l

    else:

        if (offset_l.length > offset_r.length):

            offset = offset_l

        else:

            offset = offset_r

    xA0 = x[0]
    yA0 = y[0]

    xA1 = x[-1]
    yA1 = y[-1]

    x, y = offset.coords.xy

    xB0 = x[0]
    yB0 = y[0]

    xB1 = x[-1]
    yB1 = y[-1]

    dist_00 = np.sqrt((xA0 - xB0)**2 + (yA0 - yB0)**2)
    dist_01 = np.sqrt((xA0 - xB1)**2 + (yA0 - yB1)**2)
    dist_10 = np.sqrt((xA1 - xB0)**2 + (yA1 - yB0)**2)
    dist_11 = np.sqrt((xA1 - xB1)**2 + (yA1 - yB1)**2)

    if (dist_00 + dist_11 < dist_10 + dist_01):

        offset = reverse_geom(offset)

    # print(offset_r.geom_type)
    # print('offset',offset)
    if plot_flag:
        ax.plot(*offset.xy)

    path = []

    p = Polygon([*list(medial_axis.coords), *list(offset.coords)])

    x, y = p.exterior.coords.xy
    xy_poly = [(x, y) for x, y in zip(x, y)]
    path = Path(xy_poly)

    return path


@st.cache
def cut(line):
    # Cuts a line in two at a distance from its starting point

    coords = list(line.coords)
    linePoly = Polygon(coords)
    print('length',line.length)
    print('area',linePoly.area)
    
    perimeter = line.length
    equivalent_area_perimeter = 2.0*np.pi*np.sqrt(linePoly.area/np.pi)
    print('equivalent area perimenter',equivalent_area_perimeter)
    
    if ( equivalent_area_perimeter / perimeter > 0.3 ):
    
        return line 
    
    
    nph = 200

    len_list = np.linspace(0,line.length,num=2*nph+1,endpoint=True)
    coords_uniform = []
    for Length in len_list:  
    
        ix,iy = line.interpolate(Length, normalized=False).xy
        
        coords_uniform.append((ix[0],iy[0]))
    
    line_uniform = LineString(coords_uniform)
    print('length line uniform',line_uniform.length)
     
    bbox_max = 0.0
    for i in range(nph+1):
    
        line_check = LineString(coords_uniform[i:i+nph+1])
        line_Poly = Polygon(coords_uniform[i:i+nph+1])
        print(i,line_check.length)
        print(i,line_Poly.area)
        
        bbox_area = line_Poly.area
        if bbox_area > bbox_max:
        
            line_opt = line_check
            bbox_max = bbox_area 
            print('i',i)
                        
    return line_opt            


@st.cache
def raster_to_vector2(X, Y, skeleton,fract1,fract2):

    from shapely.geometry import shape, JOIN_STYLE

    cn = ax.contour(X, Y, np.flipud(skeleton), 1)
    
    # Set a distance threshold to check if the contours are closed curves
    eps = 1e-5

    ln_max = 0.0

    for cc in cn.collections:
        paths = []
        # for each separate section of the contour line
        for pp in cc.get_paths():
            xy = []
            # for each segment of that section            
            for vv in pp.iter_segments():
                xy.append(vv[0])
                # print(vv[0])
            paths.append(np.vstack(xy))
            
            xv = [ vv[0][0] for vv in pp.iter_segments() ]
            yv = [ vv[0][1] for vv in pp.iter_segments() ]
            
            
            # Check if the contour is closed. We fit the contour with an
            # ellipse only if the contour is a closed curve.
            closed = (abs(xv[0] - xv[-1]) <
                      eps) and (abs(yv[0] - yv[-1]) < eps)

            points = [(x,y) for x,y in zip(xv,yv)]
            
            if closed and ( LineString(points).length > ln_max ):
            
                ln = LineString(points)
                poly_verts = points
                
                ln_max = ln.length
            
                print('number of points',len(xv))
            

    l1_1 = cut(ln)
    
    medial_axis = l1_1.simplify(skeleton_level, preserve_topology=True)
    
    return medial_axis
    
    
def file_selector(folder_path='.', ext='asc'):
    filenames = os.listdir(folder_path)
    filelist = []
    for file in filenames:
        if file.endswith(".asc"):
            filelist.append(file)

    selected_filename = st.selectbox('Select a file', filelist)
    return os.path.join(folder_path, selected_filename)


@st.cache
def save_netcdf(ascii_file,X,Y,slope, h_DEM , h, curv_var, top_variable, sigma=1.0):

    Pmax1, Pmin1 = surfature(X,Y,h)

    Pmax1_norm = (Pmax1 - np.nanmin(Pmax1)) / (np.nanmax(Pmax1) - np.nanmin(Pmax1))
    Pmin1_norm = - (Pmin1 - np.nanmin(Pmin1)) / (np.nanmax(Pmin1) - np.nanmin(Pmin1))

    H_elems = hessian_matrix(slope, sigma=10.0, order='xy')
    # eigenvalues of hessian matrix
    Pmax2, Pmin2 = hessian_matrix_eigvals(H_elems)

    Pmax2_norm = (Pmax2 - np.nanmin(Pmax2)) / (np.nanmax(Pmax2) - np.nanmin(Pmax2))
    Pmin2_norm = - (Pmin2 - np.nanmin(Pmin2)) / (np.nanmax(Pmin2) - np.nanmin(Pmin2))

    h_norm = ( h - np.nanmin(h) ) / (np.nanmax(h) - np.nanmin(h))

    slope_norm = -( slope - np.nanmin(slope) ) / ( np.nanmax(slope) - np.nanmin(slope) )

    # create netcdf4 file
    ncfilename = ascii_file.replace('.asc','.nc')

    ncfile = netCDF4.Dataset(ncfilename, mode='w', format='NETCDF4')

    x_dim = ncfile.createDimension('x', X.shape[1])
    y_dim = ncfile.createDimension('y', X.shape[0])
    # unlimited axis (can be appended to).
    time_dim = ncfile.createDimension('time', None)

    ncfile.title = ascii_file.replace('.asc',' analysis')

    ncfile.Conventions = "CF-1.0"
    ncfile.subtitle = "My model data subtitle"
    ncfile.anxthing = "write anxthing"

    x = ncfile.createVariable('x', np.float64, ('x', ), zlib=True)
    x.long_name = 'x dim'
    x.units = 'meters'

    y = ncfile.createVariable('y', np.float64, ('y', ), zlib=True)
    y.long_name = 'y dim'
    y.units = 'meters'

    t = ncfile.createVariable('time', np.float64, ('time', ))
    t.long_name = 'Time'
    t.units = 'seconds'

    # note: unlimited dimension is leftmost
    z = ncfile.createVariable('z', np.float64, ('time', 'y', 'x'), zlib=True)
    z.standard_name = 'flow thickness'  # this is a CF standard name
    z.units = 'meters'

   
    iter = 0
    t[iter] = 0.0
       
    x[:] = X[0, :]
    y[:] = Y[:, 0]

    z[iter, :, :] = np.flipud(h_DEM)

    # note: unlimited dimension is leftmost
    var1 = ncfile.createVariable('elevation+max.curvature1+slope', np.float64, ('time', 'y', 'x'), zlib=True)
    var1.standard_name = 'elevation+max.curvature1+slope'  # this is a CF standard name
    var1.units = ''
    top_var1 = ( h_norm + slope_norm + Pmax1_norm ) / 3.0
    var1[iter,:,:] = np.flipud( top_var1 )

    # note: unlimited dimension is leftmost
    var2 = ncfile.createVariable('elevation+max.curvature1', np.float64, ('time', 'y', 'x'), zlib=True)
    var2.standard_name = 'elevation+max.curvature1'  # this is a CF standard name
    var2.units = ''
    top_var2 = 0.5 * ( h_norm + Pmax1_norm )
    var2[iter,:,:] = np.flipud( top_var2 )

    # note: unlimited dimension is leftmost
    var3 = ncfile.createVariable('max.curvature1+slope', np.float64, ('time', 'y', 'x'), zlib=True)
    var3.standard_name = 'max.curvature1+slope'  # this is a CF standard name
    var3.units = ''
    top_var3 = 0.5 * ( slope_norm + Pmax1_norm )
    var3[iter,:,:] = np.flipud( top_var3 )

    # note: unlimited dimension is leftmost
    var4 = ncfile.createVariable('elevation+slope', np.float64, ('time', 'y', 'x'), zlib=True)
    var4.standard_name = 'elevation+slope'  # this is a CF standard name
    var4.units = ''
    top_var4 = 0.5 * ( h_norm + slope_norm )
    var4[iter,:,:] = np.flipud( top_var4 )

    # note: unlimited dimension is leftmost
    var5 = ncfile.createVariable('elevation', np.float64, ('time', 'y', 'x'), zlib=True)
    var5.standard_name = 'elevation'  # this is a CF standard name
    var5.units = ''
    top_var5 = h_norm
    var5[iter,:,:] = np.flipud( top_var5 )

    # note: unlimited dimension is leftmost
    var6 = ncfile.createVariable('max.curvature1', np.float64, ('time', 'y', 'x'), zlib=True)
    var6.standard_name = 'max.curvature1'  # this is a CF standard name
    var6.units = ''
    top_var6 = Pmax1_norm
    var6[iter,:,:] = np.flipud( top_var6 )

    # note: unlimited dimension is leftmost
    var7 = ncfile.createVariable('slope', np.float64, ('time', 'y', 'x'), zlib=True)
    var7.standard_name = 'slope'  # this is a CF standard name
    var7.units = ''
    top_var7 = slope_norm
    var7[iter,:,:] = np.flipud( slope_norm )

    # note: unlimited dimension is leftmost
    var8 = ncfile.createVariable('elevation+max.curvature2+slope', np.float64, ('time', 'y', 'x'), zlib=True)
    var8.standard_name = 'elevation+max.curvature2+slope'  # this is a CF standard name
    var8.units = ''
    top_var8 = ( h_norm + slope_norm + Pmax2_norm ) / 3.0
    var8[iter,:,:] = np.flipud( top_var8 )

    # note: unlimited dimension is leftmost
    var9 = ncfile.createVariable('elevation+max.curvature2', np.float64, ('time', 'y', 'x'), zlib=True)
    var9.standard_name = 'elevation+max.curvature2'  # this is a CF standard name
    var9.units = ''
    top_var9 = 0.5 * ( h_norm + Pmax2_norm )
    var9[iter,:,:] = np.flipud( top_var9 )

    # note: unlimited dimension is leftmost
    var10 = ncfile.createVariable('max.curvature2+slope', np.float64, ('time', 'y', 'x'), zlib=True)
    var10.standard_name = 'max.curvature2+slope'  # this is a CF standard name
    var10.units = ''
    top_var10 = 0.5 * ( slope_norm + Pmax2_norm )
    var10[iter,:,:] = np.flipud( top_var10 )

    # note: unlimited dimension is leftmost
    var11 = ncfile.createVariable('max.curvature2', np.float64, ('time', 'y', 'x'), zlib=True)
    var11.standard_name = 'max.curvature2'  # this is a CF standard name
    var11.units = ''
    top_var11 = Pmax2_norm
    var11[iter,:,:] = np.flipud( top_var11 )



    ncfile.close()


@st.cache
def detect_ridges(X,Y,slope, h, curv_var, top_variable, sigma=1.0):

    Pmax1, Pmin1 = surfature(X,Y,h)

    Pmax1_norm = (Pmax1 - np.nanmin(Pmax1)) / (np.nanmax(Pmax1) - np.nanmin(Pmax1))
    Pmin1_norm = - (Pmin1 - np.nanmin(Pmin1)) / (np.nanmax(Pmin1) - np.nanmin(Pmin1))

    H_elems = hessian_matrix(slope, sigma=10.0, order='xy')
    # eigenvalues of hessian matrix
    Pmax2, Pmin2 = hessian_matrix_eigvals(H_elems)

    Pmax2_norm = (Pmax2 - np.nanmin(Pmax2)) / (np.nanmax(Pmax2) - np.nanmin(Pmax2))
    Pmin2_norm = - (Pmin2 - np.nanmin(Pmin2)) / (np.nanmax(Pmin2) - np.nanmin(Pmin2))

    if curv_var == 'elevation':

        Pmax_norm = Pmax1_norm
        Pmin_norm = Pmin1_norm

    elif curv_var == 'slope':

        Pmax_norm = Pmax2_norm
        Pmin_norm = Pmin2_norm

    h_norm = ( h - np.nanmin(h) ) / (np.nanmax(h) - np.nanmin(h))

    slope_norm = -( slope - np.nanmin(slope) ) / ( np.nanmax(slope) - np.nanmin(slope) )
            
    print('top variable',top_variable)

    if top_variable == 'elevation+max.curvature+slope':
    
        top_var = ( h_norm + slope_norm + Pmax_norm ) / 3.0
       
    elif top_variable == 'elevation+max.curvature':
    
        top_var = 0.5 * ( h_norm + Pmax_norm )
    
    elif top_variable ==  'max.curvature+slope':

        top_var = 0.5 * (slope_norm + Pmax_norm )

    elif top_variable ==  'elevation+slope':

        top_var = 0.5 * ( h_norm + slope_norm ) 

    elif top_variable ==  'elevation':

        top_var = h_norm
        
    elif top_variable ==  'min.curvature':

        top_var = Pmin_norm

    elif top_variable ==  'max.curvature':

        top_var = Pmax_norm

    elif top_variable ==  'slope':

        top_var = slope_norm

    norm_image = cv2.normalize(top_var,None,
                                   alpha=0,
                                   beta=255,
                                   norm_type=cv2.NORM_MINMAX,
                                   dtype=cv2.CV_32F)

    norm_image = norm_image.astype(np.uint8)

    return norm_image


@st.cache
def apply_thresh(norm_image):

    # smooth the normalized image
    # blur_image = cv2.GaussianBlur(norm_image, (9, 9), 0)
    blur_image = norm_image

    # apply a threshold and create bimary image from
    # threshold (level of grey>thresh_level)
    img = cv2.threshold(blur_image, thresh_level, 255, cv2.THRESH_BINARY)[1]

    
    # remove small holes
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # img = closing

    return img


@st.cache
def compute_slope(h, delta_x, delta_y):

    # elevation partial derivatives (dh/dx and dh/dy)
    h_x, h_y = np.gradient(h, delta_x, delta_y)

    # magnitude of elevation gradient
    grad_h = np.sqrt(h_x**2 + h_y**2)

    # slope of topography in degrees
    slope = np.arctan(grad_h) * 180.0 / np.pi

    return grad_h, h_x, h_y, slope


@st.cache
def opt_connected_comp(img, h):

    # search for connected components
    num_labels, labels = cv2.connectedComponents(img)

    h_mean_opt = 0
    h_max_opt = 0

    # loop to search for connected components with largest mean elevation
    for label in range(1, num_labels):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 255

        h_mean = np.mean(h[labels == label])
        h_max = np.max(h[labels == label])

        if (h_max > h_max_opt):

            label_opt = label
            h_max_opt = h_max

    # create a raster layer with 1 for the best connected component
    binary = np.zeros_like(mask).astype(float)
    binary[labels == label_opt] = 1.0
    binary[labels != label_opt] = 0
    binary[binary == 0.0] = np.nan

    return binary


@st.cache
def find_slow(binary, h_smooth, scal_dot, val_thr):

    # create a raster layer with
    binaryL = np.zeros_like(binary).astype(bool)
    binaryL[binary == 1] = True

    DeltaN = (h_smooth[1:-1, 1:-1] < h_smooth[2:, 1:-1]).astype(bool)
    DeltaS = (h_smooth[1:-1, 1:-1] < h_smooth[0:-2, 1:-1]).astype(bool)
    DeltaW = (h_smooth[1:-1, 1:-1] < h_smooth[1:-1, 2:]).astype(bool)
    DeltaE = (h_smooth[1:-1, 1:-1] < h_smooth[1:-1, 0:-2]).astype(bool)

    DeltaNE = (h_smooth[1:-1, 1:-1] < h_smooth[2:, 0:-2]).astype(bool)
    DeltaSE = (h_smooth[1:-1, 1:-1] < h_smooth[0:-2, 0:-2]).astype(bool)
    DeltaNW = (h_smooth[1:-1, 1:-1] < h_smooth[2:, 2:]).astype(bool)
    DeltaSW = (h_smooth[1:-1, 1:-1] < h_smooth[0:-2, 2:]).astype(bool)

    print('val_thr', val_thr)

    scal_dot_thr = (scal_dot[1:-1, 1:-1] > val_thr).astype(bool)

    DeltaN = np.logical_and(DeltaN, scal_dot_thr)
    DeltaS = np.logical_and(DeltaS, scal_dot_thr)
    DeltaW = np.logical_and(DeltaW, scal_dot_thr)
    DeltaE = np.logical_and(DeltaE, scal_dot_thr)

    DeltaNE = np.logical_and(DeltaNE, scal_dot_thr)
    DeltaSE = np.logical_and(DeltaSE, scal_dot_thr)
    DeltaNW = np.logical_and(DeltaNW, scal_dot_thr)
    DeltaSW = np.logical_and(DeltaSW, scal_dot_thr)

    tic = time.perf_counter()
    np_new = np.sum(binaryL)

    for nbr in range(1500):

        binaryL[1:-1, 1:-1] += binaryL[2:, 1:-1] * DeltaN + \
            binaryL[0:-2, 1:-1] * DeltaS + binaryL[1:-1, 2:] * DeltaW + \
            binaryL[1:-1, 0:-2] * DeltaE + binaryL[2:, 2:] * DeltaNW + \
            binaryL[0:-2, 2:] * DeltaSW + binaryL[2:, 0:-2] * DeltaNE + \
            binaryL[0:-2, 0:-2] * DeltaSE

        np_old = np_new
        np_new = np.sum(binaryL)
        # print('np_new',np_new)

        if np_new == np_old:

            print('np_new', np_new)

            break

    toc = time.perf_counter()
    print(f"Loop in {toc - tic:0.4f} seconds")

    mask = np.array(binaryL * 255, dtype=np.uint8)
    img = cv2.threshold(mask, 122, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    img = opening

    # search for connected components
    num_labels, labels = cv2.connectedComponents(img)

    comp_opt = 0

    for label in range(1, num_labels):
        mask = np.zeros_like(img)
        mask[labels == label] = 1

        comp_max = np.sum(mask)

        if (comp_max > comp_opt):

            label_opt = label
            comp_opt = comp_max

    mask = np.zeros_like(img)
    mask[labels == label_opt] = 1
    img = ndimage.binary_fill_holes(mask).astype(np.uint8)

    return img


@st.cache
def compute_scal_dot(h_smooth, skeleton):

    # compute the distance (edt) in pixels of each map pixel from the skeleton.
    # for each pixel we also compute the index (i,j) of the closest
    # skeleton pixel
    edt, inds = ndimage.distance_transform_edt(skeleton == 0,
                                               return_indices=True)

    # rescale the distance in meters
    edt *= dx

    # for each pixel, we compute the vector from the closest
    # skeleton point to the pixel
    synth_x = X - X[tuple(inds)]
    synth_y = Y - Y[tuple(inds)]

    # magnitude of the vector
    synth_mag = np.sqrt(synth_x**2 + synth_y**2)

    # normalized vector
    synth_x /= synth_mag
    synth_y /= synth_mag

    # elevation partial derivatives (dh/dx and dh/dy)
    h_x, h_y = np.gradient(h_smooth, delta_x, delta_y)

    # magnitude of the gradient (=abs(tan(slope)))
    grad_h_mag = np.sqrt(h_x**2 + h_y**2)

    h_x /= grad_h_mag
    h_y /= grad_h_mag

    # negative of scalar dot between normalized vector and
    # normalized smoothed gradient. when this number=1,
    # the two vectors have the same direction. when this
    # number=-1 the two vectors have opposite direction
    scal_dot = -h_x * synth_y - h_y * synth_x

    return scal_dot, grad_h_mag


if __name__ == '__main__':

    temp_path = './temp/'

    flank_check = False

    # Check whether the specified path exists or not
    isExist = os.path.exists(temp_path)

    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(temp_path)
        print("The new directory is created!")

    ascii_file = file_selector(ext='asc')
    st.write('You selected `%s`' % ascii_file)

    [X, Y, h, x_min, x_max, delta_x, y_min, y_max,
     delta_y] = read_asc(ascii_file)

    dx = np.abs(X[1, 2] - X[1, 1])
    dy = np.abs(Y[2, 1] - Y[1, 1])

    h = np.flipud(h)
    
    h_DEM = h

    smoothing = st.sidebar.slider("Smoothing", 0, 20, 5)
    smoothing = 2*smoothing+1
    h = cv2.GaussianBlur(h, (smoothing, smoothing), 0)


    ls = LightSource(azdeg=315, altdeg=45)

    extent = [x_min, x_max, y_min, y_max]

    fig, ax = plt.subplots()

    hill_check = st.sidebar.checkbox('DEM Hillshade')

    if hill_check:

        ax.imshow(ls.hillshade(h, vert_exag=1.0, dx=delta_x, dy=delta_y),
                  cmap='gray',
                  extent=extent)

    else:

        ax.imshow(h, cmap='gray', extent=extent)

    grad_h, h_x, h_y, slope = compute_slope(h, delta_x, delta_y)

    slope = (np.maximum(0.0, np.minimum(slope, 33.0))) / 33.0

    slope_check = st.sidebar.checkbox('Slope')

    slope_opacity = st.sidebar.slider("Slope opacity", 0, 100, 50)
    slope_alpha = slope_opacity / 100.0

    if slope_check:

        ax.imshow(slope, cmap='gray', extent=extent, alpha=slope_alpha)

    st.sidebar.markdown("""---""")
    
    curv_var = st.sidebar.selectbox('Curvature variable',(
                     'elevation',
                     'slope')
                    ) 
    
    top_variable = st.sidebar.selectbox('Cone top detection variable',(
                     #'elevation+min.curvature+slope',
                     #'elevation+min.curvature',
                     #'min.curvature+slope',
                     'elevation+max.curvature+slope',
                     'elevation+max.curvature',
                     'max.curvature+slope',
                     'elevation+slope',
                     'elevation',
                     #'min.curvature',
                     'max.curvature',
                     'slope')
                    ) 



    norm_image = detect_ridges(X,Y,slope, h, curv_var, top_variable, sigma=10.0, )

    sec_der_plot_check = st.sidebar.checkbox('Slope Second Derivative Plot')

    a_opacity = st.sidebar.slider("Slope Second Derivative opacity", 0, 100,
                                  50)
    a_alpha = a_opacity / 100.0

    save_top_var_check = st.sidebar.button('NetCDF for top save')

    if save_top_var_check:
    
      save_netcdf(ascii_file,X,Y,slope, h_DEM , h, curv_var, top_variable, sigma=1.0)        

    if sec_der_plot_check:

        ax.imshow(norm_image, cmap='gray', extent=extent, alpha=a_alpha)

    st.sidebar.markdown("""---""")

    thresh_check = st.sidebar.checkbox('Threshold')

    thresh_level = st.sidebar.slider("Threshold level", 1, 255, 100)
    thresh_opacity = st.sidebar.slider("Threshold opacity", 0, 100, 50)
    thresh_alpha = thresh_opacity / 100.0

    if thresh_check:

        img = apply_thresh(norm_image)
        img_plot = np.ma.masked_where(img != 255, img)

        # ax.imshow(img_plot, cmap='gray', extent=extent, alpha=thresh_alpha)

        binary = opt_connected_comp(img, h)
        binary_plot = np.ma.masked_where(binary == 1, binary)

        # get a copy of the gray color map
        my_cmap = copy.copy(plt.cm.get_cmap('viridis'))
        
        ax.imshow(binary, cmap=my_cmap, extent=extent, alpha=thresh_alpha)

        skeleton_check = st.sidebar.checkbox('Skeleton')
        skeleton_opacity = st.sidebar.slider("Skeleton opacity", 0, 100, 50)
        skeleton_alpha = skeleton_opacity / 100.0
        
    else:
    
        skeleton_check = False    
        
       
    if skeleton_check:

        # create the skeleton of the connected component
        # the skeleton is still a raster, but reduced to
        # 1 pixel wide representation
        
        skeleton_binary = skeletonize(binary, method='lee')
        print('size',binary.shape[0],binary.shape[1],binary.shape[0]*binary.shape[1])
        print('unique',np.unique(skeleton_binary))
        for i in np.unique(skeleton_binary):
        
            print(i,np.sum(skeleton_binary==i))
            
        img = cv2.threshold(skeleton_binary, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
        
        img = np.zeros_like(skeleton_binary)
        img[skeleton_binary==0] = 1
        for i in np.unique(img):
        
            print('img',i,np.sum(img==i))
         
        from scipy import ndimage
        labeled, nr_objects = ndimage.label(img > 0.5) 
        print("Number of objects is {}".format(nr_objects))
        print('Labels',np.unique(labeled))
        # ax.imshow(labeled, extent=extent, alpha=skeleton_alpha)
        
        if nr_objects > 1:
        
            # loop to search for connected components with largest mean elevation
            label_opt = 0
            label_sum = 0
            for label in range(nr_objects+1):
        
                label_check = np.sum(labeled==label)        
                print('label,label_check',label,label_check)
            
                if ( label_check > label_sum ):
            
                    label_sum = label_check
                    label_opt = label
                
        
            img = np.zeros_like(skeleton_binary)
            img[labeled==label_opt] = 1
            # ax.imshow(img, cmap=my_cmap, extent=extent, alpha=skeleton_alpha)
        
            # remove small holes
            kernel = np.ones((5,5), np.uint8)
 
            # The first parameter is the original image,
            # kernel is the matrix with which image is
            # convolved and third parameter is the number
            # of iterations, which will determine how much
            # you want to erode/dilate a given image.
            img_dilation = cv2.dilate(img, kernel, iterations=1)
            img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
            # ax.imshow(img_erosion, cmap=my_cmap, extent=extent, alpha=skeleton_alpha)
        
            skeleton = 255 * img_erosion
    
        else: 
        
            skeleton = 255 * skeleton_binary
            pruning_size = st.sidebar.slider("Skeleton pruning size", 0, 1000, 50)
            pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(skel_img=skeleton, size=pruning_size)            
            skeleton = pruned_skeleton.astype(float)

        ax.imshow(skeleton, cmap=my_cmap, extent=extent, alpha=skeleton_alpha)


        medial_axis_check = st.sidebar.checkbox('Skeleton vector')
        skeleton_level = st.sidebar.slider("Skeleton vector simplify level", 0, 20, 5)

    else:
        
        medial_axis_check = False


    if medial_axis_check:

        # we compute a vector representation of the skeleton
        # medial_axis is a polyline defined by points
            
        medial_axis = raster_to_vector2(X, Y, skeleton,0.5,0.6)

        ax.plot(*medial_axis.xy)

        # we save the (x,y) of the points defining the polyline
        x_cnt, y_cnt = medial_axis.coords.xy

        # we create 3 lists for x,y and elevation (h) of the
        # points of the polyline. (x,y) are not pixel values,
        # but float. The elevation at (x,y) is obtained with a
        # bilinear interpolation from the pixel values
        x_top = []
        y_top = []
        h_top = []

        for (x, y) in zip(x_cnt, y_cnt):

            # indexes of the pixel lower-left
            ix = np.minimum(int(np.floor((x - x_min + 0.5 * dx) / dx)),
                                X.shape[1] - 2)
            iy = np.minimum(int(np.floor((y - y_min + 0.5 * dy) / dy)),
                                Y.shape[0] - 2)

            # indexes of the pixel top-right
            ix1 = ix + 1
            iy1 = iy + 1

            # bilinear interpolation coefficients
            alfax = (x - X[0, ix] + 0.5 * dx) / dx
            alfay = (y - Y[iy, 0] + 0.5 * dy) / dy

            # interpolated elevation
            z = alfay * (alfax * h[iy1, ix1] +
                         (1.0 - alfax) * h[iy1, ix]) + (
                             1.0 - alfay) * (alfax * h[iy, ix1] +
                                             (1.0 - alfax) * h[iy, ix])

            # this is true only if z is not nan
            if z == z:

                x_top.append(x)
                y_top.append(y)
                h_top.append(z)

        print('h_top', h_top)

        # create a 2D array with x,y,h of top_points
        data = np.c_[np.array(x_top), np.array(y_top), np.array(h_top)]
        # https://stackoverflow.com/questions/55711689/3d-plane-fitting-using-scipy-least-squares-producing-completely-wrong-fit
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]

        # C_top contains the coefficients of a plane fitting the points (x_top,y_top,h_top)
        C_top, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        print('C_top', C_top)

    # -------------- FLANK ANALYSIS --------------------

        st.sidebar.markdown("""---""")

        flank_check = st.sidebar.checkbox('Flank')
        flank_var = 'Slow'

        correction_factor = st.sidebar.slider("Scalar dot", 0.00, 1.00,
                                          0.50)

        flank_opacity = st.sidebar.slider("Flank opacity", 0, 100, 50)
        flank_alpha = flank_opacity / 100.0

    # -------------- FLANK ANALYSIS --------------------

        st.sidebar.markdown("""---""")

        buffer_check = st.sidebar.checkbox('Buffer')
        buffer_distance = st.sidebar.slider("Buffer distance", 10, 2000, 300)
        buffer_opacity = st.sidebar.slider("Buffer opacity", 0, 100, 50)
        buffer_alpha = buffer_opacity / 100.0

    else:
    
        flank_check = False
        buffer_check = False

    if flank_check:

        h_smooth = cv2.GaussianBlur(h, (5, 5), 0)

        scal_dot, grad_h_mag = compute_scal_dot(h_smooth, skeleton)

        if flank_var == 'Normal':

            # rescale the number from [-1;1] to [0;1]
            scal_dot = 0.5 * (1.0 + scal_dot)

        elif flank_var == 'Corrected':

            # rescale the number from [-1;1] to [0;1]
            scal_dot = 0.5 * (1.0 + scal_dot)

            # compute the slope in degrees from smoothed gradient
            slope = np.arctan(grad_h_mag) * 180.0 / np.pi

            # set min slope to 0 and max to 33 degrees
            slope = (np.maximum(0.0, np.minimum(slope, 33.0))) / 33.0

            # define a variable accounting for both the scalar dot
            # and the slope
            exponent = correction_factor
            scal_dot = scal_dot**(1.0 - exponent) * slope**exponent

        elif flank_var == 'Slow':

            img = find_slow(binary, h_smooth, scal_dot, correction_factor)

            scal_dot = img * scal_dot

        ax.imshow(scal_dot, cmap='gray', extent=extent, alpha=flank_alpha)
        scal_dot_scaled = 255 * scal_dot


        if buffer_check:
        
            mask_check = True
        
                                   
    else:
    
        mask_check = False

    if buffer_check:
    
        # compute buffer area (points within a fixed distance from medial axis)
        path = offset_path(medial_axis, buffer_distance, True)

        pixel_coordinates = np.c_[X.ravel(), Y.ravel()]

        # find points within path
        img_buffer = np.flipud(
            path.contains_points(pixel_coordinates).reshape(
                X.shape[0], X.shape[1]))
    
        ax.imshow(img_buffer, cmap='gray', extent=extent, alpha=buffer_alpha)
        
        if flank_check:
        
            mask_check = True
        
    else:
    
        mask_check = False   
        
   
    flank_thr_check = False
    
    if mask_check:
    
        # -------------- FLANK THRESHOLD --------------------
        st.sidebar.markdown("""---""")

        flank_thr_check = st.sidebar.checkbox('Mask save')
        flank_radio = st.sidebar.radio('Select image:',
                                   ['Buffer', 'Flank', 'Intersection'])

        mask_opacity = st.sidebar.slider("Mask opacity", 0, 100, 50)
        mask_alpha = mask_opacity / 100.0
        

    if flank_thr_check:
        
        img_01 = img
                
        img_2 = img_buffer        

        if flank_radio == 'Buffer':

            mask = img_2

        elif flank_radio == 'Flank':

            mask = img_01

        else:

            # find intersection of buffer and flank
            img = np.logical_and(img_01, img_2)
            img = img.astype('uint8')

            num_labels, labels = cv2.connectedComponents(img)

            comp_opt = 0

            for label in range(1, num_labels):
                mask = np.zeros_like(img)
                mask[labels == label] = 1

                comp_max = np.sum(mask)

                if (comp_max > comp_opt):

                    label_opt = label
                    comp_opt = comp_max

            mask = np.zeros_like(img)
            mask[labels == label_opt] = 1
            img = ndimage.binary_fill_holes(mask).astype(np.uint8)

            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = closing

            mask = img

        ax.imshow(mask, cmap='gray', extent=extent, alpha=mask_alpha)

        # Save mask on ascii raster file
        header = "ncols     %s\n" % h.shape[1]
        header += "nrows    %s\n" % h.shape[0]
        header += "xllcenter " + str(np.amin(X)) + "\n"
        header += "yllcenter " + str(np.amin(Y)) + "\n"
        header += "cellsize " + str(np.abs(X[1, 2] - X[1, 1])) + "\n"
        header += "NODATA_value -9999\n"

        output_full = ascii_file.replace('.asc', '_mask.asc')

        print('mask min max', np.nanmin(img), np.nanmax(img))

        np.savetxt(temp_path + output_full,
                   img,
                   header=header,
                   fmt='%1.5f',
                   comments='')

        with open(temp_path + output_full) as f:
            st.sidebar.download_button('Download ' + output_full,
                                       f,
                                       file_name=output_full)

        # -------------- FLANK VOLUME --------------------
        st.sidebar.markdown("""---""")

        volume_check = st.sidebar.checkbox('Volume analysis')

    else:
    
        volume_check = False

    if volume_check:

        fig_c, ax_c = plt.subplots()
        
        cn = ax_c.contour(X, Y, np.flipud(mask), 1)

        base_len = 0

        for cc in cn.collections:
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                xy = []
                # for each segment of that section            
                for vv in pp.iter_segments():
                    xy.append(vv[0])
                    # print(vv[0])
                paths.append(np.vstack(xy))
            
                xv = [ vv[0][0] for vv in pp.iter_segments() ]
                yv = [ vv[0][1] for vv in pp.iter_segments() ]
            
                if len(xv) > base_len:
                
                    x_cnt = xv
                    y_cnt = yv
                    base_len = len(xv)
        
        
        line_cnt = LineString(zip(x_cnt, y_cnt))
        line_cnt = line_cnt.simplify(1.0, preserve_topology=False)

        x_cnt, y_cnt = line_cnt.coords.xy
        x_base = []
        y_base = []
        h_base = []

        for (x, y) in zip(x_cnt, y_cnt):

            dist = medial_axis.distance(Point(x, y))

            if (dist > 10.0):

                ix = np.minimum(int(np.floor((x - x_min) / dx)),
                                X.shape[1] - 2)
                iy = np.minimum(int(np.floor((y - y_min) / dx)),
                                Y.shape[0] - 2)

                ix1 = ix + 1
                iy1 = iy + 1

                alfax = (x - X[0, ix]) / dx
                alfay = (y - Y[iy, 0]) / dy

                z = alfay * (alfax * h[iy1, ix1] +
                             (1.0 - alfax) * h[iy1, ix]) + (
                                 1.0 - alfay) * (alfax * h[iy, ix1] +
                                                 (1.0 - alfax) * h[iy, ix])

                if z == z:

                    x_base.append(x)
                    y_base.append(y)
                    h_base.append(z)
        
        ax.plot(x_base,y_base,'.r')
        
        x_base_avg = np.mean(x_base)         
        y_base_avg = np.mean(y_base)
        
        x_base_rel = x_base - x_base_avg
        y_base_rel = y_base - y_base_avg

        # print('h_base',h_base)

        # create a 2D array with x,y,h of top_points
        data = np.c_[np.array(x_base_rel), np.array(y_base_rel), np.array(h_base)]
        # https://stackoverflow.com/questions/55711689/3d-plane-fitting-using-scipy-least-squares-producing-completely-wrong-fit
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        print('C', C)

        # evaluate the elevation on the plane for the ellipse points
        # Z_ell = C[0] * ell_coord[:, 0] + C[1] * ell_coord[:, 1] + C[2]

        h_base = C[0] * ( X - x_base_avg ) + C[1] * ( Y - y_base_avg ) + C[2]

        h_mask = np.ma.masked_where(mask > 0, h - h_base)

        flank_vol = dx * dy * np.nansum(h_mask)

        print('flank_vol', flank_vol)

        st.sidebar.write('Flank volume =', flank_vol, 'm3')
        st.sidebar.write('Base plane z =', C[2],'+',C[0],'x_rel +',C[1],'y_rel')

    # -------------- SLOPE ANALYSIS --------------------

    if flank_thr_check:
    
        st.sidebar.markdown("""---""")

        slope_check = st.sidebar.checkbox('Slope analysis')
        smooth_level = st.sidebar.slider("Slope smoothing level", 0, 20, 5)

    else:
    
        slope_check = False

    if slope_check:

        # aspect from topography
        h_x_filtered = filters.gaussian(h_x, smooth_level)
        h_y_filtered = filters.gaussian(h_y, smooth_level)

        grad_h_mag = np.sqrt(h_x_filtered**2 + h_y_filtered**2)
        slope = np.arctan(grad_h_mag) * 180.0 / np.pi
        masked_slope = ma.masked_array(slope, mask == 0)
        slope_flat = masked_slope[masked_slope.mask == False].flatten()

        dist, inds = ndimage.distance_transform_edt(skeleton == 0,
                                                    return_indices=True)

        dist *= dx
        masked_dist = ma.masked_array(dist, mask == 0)
        dist_flat = masked_dist[masked_dist.mask == False].flatten()

        fig_slope, ax_slope = plt.subplots()
        # plot the cumulative histogram
        n_bins = 50
        ax_slope.hist(slope_flat,
                      n_bins,
                      density=True,
                      histtype='step',
                      cumulative=True,
                      label='Empirical')

        ax_slope.set_xlabel('slope [degrees]')
        ax_slope.set_ylabel('cumulative distribution')

        df = pd.DataFrame({'dist': dist_flat, 'slope': slope_flat})
        filename = ascii_file.replace('.asc', '_slopes.csv')
        df.to_csv(temp_path + filename, index=False)
        with open(temp_path + filename) as f:
            # Defaults to 'text/plain'
            st.sidebar.download_button('Download ' + filename,
                                       f,
                                       file_name=filename)

        n_dist = 10
        # create the bins (intervals) for the sectors based on the distance
        dist_bins = np.linspace(np.min(dist_flat), np.max(dist_flat),
                                n_dist + 1)
        dist_half = 0.5 * (dist_bins[0:-1] + dist_bins[1:])

        print('dist_bins', dist_bins)

        # group the dataframe elements in sub-domains by using partition
        groups = df.groupby([pd.cut(df.dist, dist_bins)])

        # compute the mean slope in each sub-domain. It is a 2D numpy array
        slp_mean = np.array(groups['slope'].mean())
        slp_std = np.array(groups['slope'].std())

        print('slp_mean', slp_mean)
        print('slp_std', slp_std)

        df = pd.DataFrame({
            'distance': dist_half,
            'Mean Slope': slp_mean,
            'Std slope': slp_std
        })

        filename = ascii_file.replace('.asc', '_dist_slopes.csv')
        df.to_csv(temp_path + filename, index=False)
        with open(temp_path + filename) as f:
            # Defaults to 'text/plain'
            st.sidebar.download_button('Download ' + filename,
                                       f,
                                       file_name=filename)

        fig_slope2, ax_slope2 = plt.subplots()
        ax_slope2.errorbar(dist_half, slp_mean, yerr=slp_std)
        ax_slope2.set_xlabel('distance from top [m]')
        ax_slope2.set_ylabel('slope [degrees]')

    # -------------- SYNTHETIC CONE --------------------

    if flank_thr_check:
    
        st.sidebar.markdown("""---""")

        synth_check = st.sidebar.checkbox('Synthetic cone')
        synth_opacity = st.sidebar.slider("Synthetic cone opacity", 0, 100, 50)
        synth_alpha = synth_opacity / 100.0

        cr_slope = st.sidebar.number_input("Slope angle",
                                       min_value=5,
                                       max_value=40,
                                       value=33,
                                       step=1)
        
    else:
    
        synth_check = False


    if synth_check:

        edt, inds = ndimage.distance_transform_edt(skeleton == 0,
                                                   return_indices=True)

        edt *= dx

        
        # width_cone = h_cone / np.tan(np.radians(cr_slope))
        # edt = np.minimum(edt, width_cone)

        # synth_cone = -(edt - width_cone) / width_cone * h_cone

        # print('Volume = ',np.sum(synth_cone))
        
        h_temp = h[tuple(inds)] - edt * np.tan(np.radians(cr_slope))
        smoothing = 39
        h_temp = cv2.GaussianBlur(h_temp, (smoothing, smoothing), 0)

        h0 = 0.0

        synth_cone = np.maximum(h_base,h_temp+h0)
        synth_cone = np.ma.masked_where(mask == 0, synth_cone)
        synth_vol0 = dx * dy * np.nansum(synth_cone-h_base)
        print('synth_vol0', synth_vol0)

        if synth_vol0 < flank_vol:
        
            h2 = 300.0
            synth_cone = np.maximum(h_base,h_temp+h2)
            synth_cone = np.ma.masked_where(mask == 0, synth_cone)
            synth_vol2 = dx * dy * np.nansum(synth_cone-h_base)
            print('synth_vol2', synth_vol2)

        for i in range(20):
        
            h1 = 0.5 * ( h0+h2 )
            synth_cone = np.maximum(h_base,h_temp+h1)
            synth_cone = np.ma.masked_where(mask == 0, synth_cone)
            synth_vol1 = dx * dy * np.nansum(synth_cone-h_base)
            print('h1',h1)
            print('synth_vol1', synth_vol1)

            if ( synth_vol1 < flank_vol ):
            
                h0 = h1
                
            else:
            
                h2 = h1
        
        synth_cone = np.maximum(h_base,h_temp+h1)
                           
        # synth_cone = np.maximum(h_base,h_base + h_cone - edt * np.tan(np.radians(cr_slope)))



        ax.imshow(ls.hillshade(synth_cone,
                               vert_exag=1.0,
                               dx=delta_x,
                               dy=delta_y),
                  cmap='gray',
                  extent=extent,
                  alpha=synth_alpha)

        # synth_mask = np.ma.masked_where(mask > 0, synth_cone)

        # synth_vol = dx * dy * np.nansum(synth_cone-h_base)

        # print('synth_vol', synth_vol)

        st.sidebar.write('Synthetic cone flank volume =', synth_vol1, 'm3')

        df = pd.DataFrame({
            'ascii_file': [ascii_file],
            'Top variable': [top_variable],
            'Threshold level': [thresh_level],
            'Pruning size': [pruning_size],
            'Skeleton simplify level': [skeleton_level],
            'correction_factor': [correction_factor],
            'buffer_distance': [buffer_distance],
            'flank_radio': [flank_radio],
            'Slope angle': [cr_slope]
        })

        filename = ascii_file.replace('.asc', '.csv')
        df.to_csv(temp_path + filename, index=False)
        with open(temp_path + filename) as f:
            # Defaults to 'text/plain'
            st.sidebar.download_button('Download ' + filename,
                                       f,
                                       file_name=filename)

        # Save synthetictopography on ascii raster file
        header = "ncols     %s\n" % h.shape[1]
        header += "nrows    %s\n" % h.shape[0]
        header += "xllcenter " + str(np.amin(X)) + "\n"
        header += "yllcenter " + str(np.amin(Y)) + "\n"
        header += "cellsize " + str(np.abs(X[1, 2] - X[1, 1])) + "\n"
        header += "NODATA_value -9999\n"

        output_full = ascii_file.replace('.asc', '_synth.asc')

        synth_cone_asc = synth_cone
        synth_cone_asc[np.isnan(synth_cone_asc)] = 0
        h_synth = h_base + synth_cone_asc

        np.savetxt(temp_path + output_full,
                   h_synth,
                   header=header,
                   fmt='%1.5f',
                   comments='')

        with open(temp_path + output_full) as f:
            # Defaults to 'text/plain'
            st.sidebar.download_button('Download ' + output_full,
                                       f,
                                       file_name=output_full)

    # -------------- SAVE USER INPUT --------------------

    for filename in os.listdir(temp_path):
        file_path = os.path.join(temp_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    st.pyplot(fig)
    if flank_check and skeleton_check and slope_check:
        st.pyplot(fig_slope)
        st.pyplot(fig_slope2)
