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

from matplotlib.patches import Ellipse
from matplotlib.colors import LightSource
from scipy import interpolate
import scipy.linalg
import pandas as pd

import numpy as np
import copy
from scipy import ndimage

import shapefile
from shapely.geometry import Point, Polygon, LineString, MultiPoint, MultiLineString, LinearRing
from shapely.ops import linemerge
from shapely.affinity import translate

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import geometry, ops
from matplotlib.path import Path
import numpy.ma as ma

# import whitebox
import netCDF4

import scipy.optimize


@st.cache
def fitEllipse(x, y, method):

    # fit a polyline with an ellipse
    # INPUT:
    # - x x-coordinates of the points of the polyline
    # - y y-coordinates of the points of the polyline
    # OUTPUT:
    # - cx,cy coordinates of the ellipse center
    # - a,b semiaxis of the ellipse
    # - angle angle between the major semiaxes and the x-axis

    x = x[:, None]
    y = y[:, None]

    D = np.hstack([x * x, x * y, y * y, x, y, np.ones(x.shape)])
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))

    if method == 1:
        n = np.argmax(np.abs(E))
    else:
        n = np.argmax(E)
    a = V[:, n]

    # -------------------Fit ellipse-------------------
    b, c, d, f, g, a = a[1] / 2., a[2], a[3] / 2., a[4] / 2., a[5], a[0]
    num = b * b - a * c
    cx = (c * d - b * f) / num
    cy = (a * f - b * d) / num

    angle = 0.5 * np.arctan(2 * b / (a - c)) * 180 / np.pi
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) *
                                                                  (a - c))) -
                               (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) *
                                                                  (a - c))) -
                               (c + a))
    a = np.sqrt(abs(up / down1))
    b = np.sqrt(abs(up / down2))

    return cx, cy, a, b, angle

@st.cache
def plane_fit(polyline, X, Y, h):
    """Fit a 2D shapely LineString with a planar surface

    Parameters
    ----------
    polyline : shapely LineString
        2D LineString defining the (x,y) coordinates of 
        the polyline points
    X : float numpy array
        numpy array of topography UTM x-coordiantes
    Y : float numpy array
        numpy array of topography UTM y-coordiantes
    h : float numpy array
        numpy array of topography elevation

    Returns
    -------
    C : float array
        coefficients of linear fit C[0]*x+C[1]*y+C[2]
    h_plane : float numpy array
        numpy array of linear fit elevation

    """

    len_list = np.linspace(0, polyline.length, num=20, endpoint=True)

    # list of 20 points sampled uniformly (based on length) from
    # the polyline defining the cone top
    coords_uniform = []
    for Length in len_list:

        ix, iy = polyline.interpolate(Length, normalized=False).xy

        coords_uniform.append((ix[0], iy[0]))

    line_uni = LineString(coords_uniform)

    # x and y coordinates of the points
    x_uni, y_uni = line_uni.coords.xy

    x_poly = []
    y_poly = []
    h_poly = []

    for (x, y) in zip(x_uni, y_uni):

        dist = polyline.distance(Point(x, y))

        if (dist < 10.0):

            # bi-linear interpolation of elevation h from X,Y grid points
            # to (x,y)

            ix = np.minimum(int(np.floor((x - x_min) / dx)), X.shape[1] - 2)
            iy = np.minimum(int(np.floor((y - y_min) / dx)), Y.shape[0] - 2)

            ix1 = ix + 1
            iy1 = iy + 1

            alfax = (x - X[0, ix]) / dx
            alfay = (y - Y[iy, 0]) / dy

            z = alfay * (alfax * h[iy1, ix1] + (1.0 - alfax) * h[iy1, ix]) + (
                1.0 - alfay) * (alfax * h[iy, ix1] + (1.0 - alfax) * h[iy, ix])

            if z == z:

                x_poly.append(x)
                y_poly.append(y)
                h_poly.append(z)

    ax.plot(x_poly, y_poly, '.r')

    # the linear fit algorithm works better for values of the
    # coordinates close to zero. for this reason we translate
    # the coordinates by using the mean coordinates of the
    # points of cone top: x_poly_avg,y_poly_avg

    x_poly_avg = np.mean(x_poly)
    y_poly_avg = np.mean(y_poly)
    print('x_base_avg', x_poly_avg)
    print('y_base_avg', y_poly_avg)

    x_poly_rel = x_poly - x_poly_avg
    y_poly_rel = y_poly - y_poly_avg

    # create a 2D array with x,y,h of top_points
    data = np.c_[np.array(x_poly_rel), np.array(y_poly_rel), np.array(h_poly)]
    # https://stackoverflow.com/questions/55711689/3d-plane-fitting-using-scipy-least-squares-producing-completely-wrong-fit
    A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

    print('C', C)

    # evaluate the elevation on the base plane for gridpoints
    h_plane = C[0] * (X - x_poly_avg) + C[1] * (Y - y_poly_avg) + C[2]

    return C, h_plane


@st.cache
def chaikins_corner_cutting(coords, refinements=5):
    """Chaikin's corner cutting algorithm. Smooths a polyline 
    by iteratively replacing every point by two new points: 
    one 1/4 of the way to the next point and one 1/4 of 
    the way to the previous point.

    Source: https://stackoverflow.com/questions/47068504/where-to-find-python-implementation-of-chaikins-corner-cutting-algorithm


    Parameters
    ----------
    coords : float list of lists
        list of points, defined by the x and y coordinated
    refinement : int
        number of refinements for the Chaikin's algorithm

    Returns
    -------
    coords : numnpy float array
        coordinates of the points defining the smoother
        polyline

    """

    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def monoExp(x, m, t, b):
    """Exponential function used for the fitting. The 
    fitting function has an horizontal asymptot y=b
    and intersect x=0 at y=m+b. The values of t controls
    how fast the exponential approaches the horizontal
    asymptot.

    Parameters
    ----------
    x : float 
        x coordinate
    t : float
        exponential function parameter    
    m : float
        exponential function parameter    
    b : float
        exponential function parameter    

    Returns
    -------
    y : float
        value of the exponential function at x

    """

    y = m * np.exp(-t * x) + b

    return y


@st.cache
def surfature(X, Y, Z):
    """Curvature of Z=f(X,Y).
    Igor V Florinsky
    An illustrated introduction to general geomorphometry
    Progress in Physical Geography
    2017, Vol. 41(6) 723–752

    Parameters
    ----------
    X : float numpy array
        numpy array of topography UTM x-coordiantes
    Y : float numpy array
        numpy array of topography UTM y-coordiantes
    z : float numpy array
        numpy array of topography elevation  

    Returns
    -------
    Pmax : float numpy array
        maximum curvature
    Pmin : float numpy array
        minimum curvature

    """
    cellsize = np.abs(X[1, 2] - X[1, 1])

    # first order local derivatives
    gx, gy = np.gradient(Z, cellsize, cellsize)

    # second order local derivatives
    gxx, gxy1 = np.gradient(gx, cellsize, cellsize)

    # second order local derivatives
    gxy2, gyy = np.gradient(gy, cellsize, cellsize)

    # mixed second order average local derivatives
    gxy = 0.5 * (gxy1 + gxy2)
    gyx = gxy

    # Eq. 2 from Florinsky 2017, "An illustrated introduction to general geomorphometry"
    # Computer Vision
    p = gx
    q = gy
    r = gxx
    t = gyy
    s = gxy

    # Eq. 21 from Florinsky 2017, "An illustrated introduction to general geomorphometry"
    # Computer Vision
    H = - ((1.0+q**2)*r - 2.0*p*q*s + (1.0 + p**2) * t) / \
        (2.0 * np.sqrt((1.0+p**2+q**2)**3))

    # Eq. 22 from Florinsky 2017, "An illustrated introduction to general geomorphometry"
    # Computer Vision
    K = (r * t - s**2) / (1.0 + p**2 + q**2)**2

    # Eq. 23 from Florinsky 2017, "An illustrated introduction to general geomorphometry"
    # Computer Vision
    M = np.sqrt(H**2 - K)

    # Principal Curvatures

    # Eq. 20 from Florinsky 2017, "An illustrated introduction to general geomorphometry"
    # Computer Vision
    Pmax = H + M

    # Eq. 19 from Florinsky 2017, "An illustrated introduction to general geomorphometry"
    # Computer Vision
    Pmin = H - M

    # n = 5
    # m = 1
    # Pmin = np.sign(Pmin)*np.log(1.0+10**(n*m)*np.abs(Pmin))
    # Pmax = np.sign(Pmax)*np.log(1.0+10**(n*m)*np.abs(Pmax))

    return Pmax, Pmin


@st.cache
def read_asc(ascii_file):
    """Read DEM in .asc format 
    Read a DEM in ESRII ascci format and UTM coordinated

    Parameters
    ----------
    ascii_file : string
        name of file to read
    Returns
    -------
    X : float numpy array
        array of grid y UTM coordinates
    Y : float numpy array
        array of grid x UTM coordinates
    h : float numpy array
        array of topography elevation    
    x_min : float
        minimum x of grid points    
    x_max : float
        maximum x of grid points   
    delta_x : float
        size of grid cells in x-direction    
    y_min : float
        minimum y of grid points        
    y_max : float
        minimum y of grid points        
    delta_y : float
        size of grid cells in y-direction

    """
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
def offset_path(skeleton_vector, offset, plot_flag):
    """Return a matplotlib at a distance from the cone
    top and in the direction of the flank

    Parameters
    ----------
    skeleton_vector : shapely LineString
        2D LineString defining the (x,y) coordinates of 
        the cone top
    offset: float 
        offset distance
    plot_flag : logical
        logical to plot the offset path

    Returns
    -------
    path : matplotlib Path
        offset path on the flank of the cone 

    """

    # offset paths on left and right of the cone top
    # we do not know a priori which one we is on the flank side
    offset_l = skeleton_vector.parallel_offset(offset, 'left', join_style=1)
    offset_r = skeleton_vector.parallel_offset(offset, 'right', join_style=1)
   
    print('offset_l',offset_l)
    print('offset_r',offset_r)

    
    if offset_l.geom_type == 'MultiLineString':

        offset = offset_r

    elif offset_r.geom_type == 'MultiLineString':

        offset = offset_l

    else:

        if (offset_l.length > offset_r.length):

            offset = offset_l

        else:

            offset = offset_r

    print('skeleton_vector',skeleton_vector.geom_type)
    print('offset',offset.geom_type)
    print(offset)

    xs, ys = skeleton_vector.coords.xy

    xA0 = xs[0]
    yA0 = ys[0]

    xA1 = xs[-1]
    yA1 = ys[-1]

    xo, yo = offset.coords.xy

    xB0 = xo[0]
    yB0 = yo[0]

    xB1 = xo[-1]
    yB1 = yo[-1]
    
    dist_00 = np.sqrt((xA0 - xB0)**2 + (yA0 - yB0)**2)
    dist_01 = np.sqrt((xA0 - xB1)**2 + (yA0 - yB1)**2)
    dist_10 = np.sqrt((xA1 - xB0)**2 + (yA1 - yB0)**2)
    dist_11 = np.sqrt((xA1 - xB1)**2 + (yA1 - yB1)**2)

    if (dist_00 + dist_11 < dist_10 + dist_01):

        offset = reverse_geom(offset)
        xo, yo = offset.coords.xy

    if plot_flag:
        ax.plot(*offset.xy)

    if skeleton_vector.geom_type == 'LinearRing':

        xo.append(xo[0])    
        yo.append(yo[0])   

    p = Polygon([(x, y) for x, y in zip(xs+xo, ys+yo)])

    x, y = p.exterior.coords.xy
    xy_poly = [(x, y) for x, y in zip(x, y)]
    path = Path(xy_poly)
 
    return path


@st.cache
def cut(line):
    # Cuts a closed line in two parts

    coords = list(line.coords)
    linePoly = Polygon(coords)
    print('length', line.length)
    print('area', linePoly.area)

    perimeter = line.length
    equivalent_area_perimeter = 2.0 * np.pi * np.sqrt(linePoly.area / np.pi)
    print('equivalent area perimenter', equivalent_area_perimeter)

    if (equivalent_area_perimeter / perimeter > 0.3):

        return LinearRing(coords), True

    nph = 200

    len_list = np.linspace(0, line.length, num=2 * nph + 1, endpoint=True)
    coords_uniform = []
    for Length in len_list:

        ix, iy = line.interpolate(Length, normalized=False).xy

        coords_uniform.append((ix[0], iy[0]))

    line_uniform = LineString(coords_uniform)
    print('length line uniform', line_uniform.length)

    bbox_max = 0.0
    for i in range(nph + 1):

        line_check = LineString(coords_uniform[i:i + nph + 1])
        line_Poly = Polygon(coords_uniform[i:i + nph + 1])
        print(i, line_check.length)
        print(i, line_Poly.area)

        bbox_area = line_Poly.area
        if bbox_area > bbox_max:

            line_opt = line_check
            bbox_max = bbox_area
            print('i', i)

    return line_opt, False


@st.cache
def raster_to_vector(X, Y, skeleton):

    from shapely.geometry import shape, JOIN_STYLE

    # create contour lines from the skeleton image
    # the contour should be a close line
    cn = ax.contour(X, Y, skeleton, 1, alpha=0)

    # Set a distance threshold to check if the contours are closed curves
    eps = 1e-5

    ln_max = 0.0
    
    # loop over the contours to take the longest close one
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

            xv = [vv[0][0] for vv in pp.iter_segments()]
            yv = [vv[0][1] for vv in pp.iter_segments()]

            # Check if the contour is closed.
            closed = (abs(xv[0] - xv[-1]) < eps) and (abs(yv[0] - yv[-1]) <
                                                      eps)
                                                      
            if closed:
            
                xv[-1] = xv[0]
                yv[-1] = yv[0]                                          

            points = [(x, y) for x, y in zip(xv, yv)]

            if closed and (LineString(points).length > ln_max):

                ln = LineString(points)
                poly_verts = points

                ln_max = ln.length

                print('number of points', len(xv))

    # if the skeleton is not closed cut the contour in two 
    # parts and take one of the two
    skeleton_vector, closed = cut(ln)

    return skeleton_vector, closed

@st.cache
def improve_vector(skeleton_vector,closed,skeleton_level,skeleton_smoothing_level):

    if skeleton_level > 0:
    
        skeleton_vector = skeleton_vector.simplify(skeleton_level, preserve_topology=True)
                
                
    if skeleton_smoothing_level > 0:

        x, y = skeleton_vector.coords.xy

        coords = np.array([[x, y] for x, y in zip(x, y)])

        coords = chaikins_corner_cutting(coords,
                                         refinements=skeleton_smoothing_level)

        if closed:
        
            skeleton_vector = LinearRing(coords)
            
        else:
            
            skeleton_vector = LineString(coords)


    if ( skeleton_range[0] > 0 ) or ( skeleton_range[1] < 100 ): 

        x, y = skeleton_vector.coords.xy

        line = []

        for i in range(len(x) - 1):

            coords = np.array([[x, y] for x, y in zip(x[i:], y[i:])])

            partial_axis = LineString(coords)

            line.append(partial_axis.length)

        line.append(0.0)
        line = np.array(line)
        line = line / max(line) * 100.0
        idx1 = (np.abs(line - skeleton_range[0])).argmin()
        idx0 = (np.abs(line - skeleton_range[1])).argmin()

        print(idx0, idx1)

        coords = np.array([[x, y] for x, y in zip(x[idx0:idx1], y[idx0:idx1])])

        skeleton_vector = LineString(coords)
        closed = False

    return skeleton_vector, closed


def file_selector(folder_path='.', ext='asc'):
    filenames = os.listdir(folder_path)
    filelist = []
    for file in filenames:
        if file.endswith(".asc"):
            filelist.append(file)

    selected_filename = st.selectbox('Select a file', filelist)
    return os.path.join(folder_path, selected_filename)


@st.cache
def save_netcdf(ascii_file,
                X,
                Y,
                slope,
                h_DEM,
                h,
                curv_var,
                curvature_variable,
                sigma=1.0):

    Pmax1, Pmin1 = surfature(X, Y, h)

    Pmax1_norm = (Pmax1 - np.nanmin(Pmax1)) / \
        (np.nanmax(Pmax1) - np.nanmin(Pmax1))
    Pmin1_norm = - (Pmin1 - np.nanmin(Pmin1)) / \
        (np.nanmax(Pmin1) - np.nanmin(Pmin1))

    H_elems = hessian_matrix(slope, sigma=10.0, order='xy')
    # eigenvalues of hessian matrix
    Pmax2, Pmin2 = hessian_matrix_eigvals(H_elems)
    
    Pmax2, Pmin2 = surfature(X, Y, slope)
    

    Pmax2_norm = (Pmax2 - np.nanmin(Pmax2)) / \
        (np.nanmax(Pmax2) - np.nanmin(Pmax2))
    Pmin2_norm = - (Pmin2 - np.nanmin(Pmin2)) / \
        (np.nanmax(Pmin2) - np.nanmin(Pmin2))

    h_norm = (h - np.nanmin(h)) / (np.nanmax(h) - np.nanmin(h))

    slope_norm = -(slope - np.nanmin(slope)) / \
        (np.nanmax(slope) - np.nanmin(slope))

    # create netcdf4 file
    ncfilename = ascii_file.replace('.asc', '.nc')

    ncfile = netCDF4.Dataset(ncfilename, mode='w', format='NETCDF4')

    x_dim = ncfile.createDimension('x', X.shape[1])
    y_dim = ncfile.createDimension('y', X.shape[0])
    # unlimited axis (can be appended to).
    time_dim = ncfile.createDimension('time', None)

    ncfile.title = ascii_file.replace('.asc', ' analysis')

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

    z[iter, :, :] = h_DEM

    # note: unlimited dimension is leftmost
    var1 = ncfile.createVariable('elevation+max.curvature1+slope',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var1.standard_name = 'elevation+max.curvature1+slope'  # this is a CF standard name
    var1.units = ''
    top_var1 = (h_norm + slope_norm + Pmax1_norm) / 3.0
    var1[iter, :, :] = top_var1

    # note: unlimited dimension is leftmost
    var2 = ncfile.createVariable('elevation+max.curvature1',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var2.standard_name = 'elevation+max.curvature1'  # this is a CF standard name
    var2.units = ''
    top_var2 = 0.5 * (h_norm + Pmax1_norm)
    var2[iter, :, :] = top_var2

    # note: unlimited dimension is leftmost
    var3 = ncfile.createVariable('max.curvature1+slope',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var3.standard_name = 'max.curvature1+slope'  # this is a CF standard name
    var3.units = ''
    top_var3 = 0.5 * (slope_norm + Pmax1_norm)
    var3[iter, :, :] = top_var3

    # note: unlimited dimension is leftmost
    var4 = ncfile.createVariable('elevation+slope',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var4.standard_name = 'elevation+slope'  # this is a CF standard name
    var4.units = ''
    top_var4 = 0.5 * (h_norm + slope_norm)
    var4[iter, :, :] = top_var4

    # note: unlimited dimension is leftmost
    var5 = ncfile.createVariable('elevation',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var5.standard_name = 'elevation'  # this is a CF standard name
    var5.units = ''
    top_var5 = h_norm
    var5[iter, :, :] = top_var5

    # note: unlimited dimension is leftmost
    var6 = ncfile.createVariable('max.curvature1',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var6.standard_name = 'max.curvature1'  # this is a CF standard name
    var6.units = ''
    top_var6 = Pmax1_norm
    var6[iter, :, :] = top_var6

    # note: unlimited dimension is leftmost
    var7 = ncfile.createVariable('slope',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var7.standard_name = 'slope'  # this is a CF standard name
    var7.units = ''
    top_var7 = slope_norm
    var7[iter, :, :] = slope_norm

    # note: unlimited dimension is leftmost
    var8 = ncfile.createVariable('elevation+max.curvature2+slope',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var8.standard_name = 'elevation+max.curvature2+slope'  # this is a CF standard name
    var8.units = ''
    top_var8 = (h_norm + slope_norm + Pmax2_norm) / 3.0
    var8[iter, :, :] = top_var8

    # note: unlimited dimension is leftmost
    var9 = ncfile.createVariable('elevation+max.curvature2',
                                 np.float64, ('time', 'y', 'x'),
                                 zlib=True)
    var9.standard_name = 'elevation+max.curvature2'  # this is a CF standard name
    var9.units = ''
    top_var9 = 0.5 * (h_norm + Pmax2_norm)
    var9[iter, :, :] = top_var9

    # note: unlimited dimension is leftmost
    var10 = ncfile.createVariable('max.curvature2+slope',
                                  np.float64, ('time', 'y', 'x'),
                                  zlib=True)
    var10.standard_name = 'max.curvature2+slope'  # this is a CF standard name
    var10.units = ''
    top_var10 = 0.5 * (slope_norm + Pmax2_norm)
    var10[iter, :, :] = top_var10

    # note: unlimited dimension is leftmost
    var11 = ncfile.createVariable('max.curvature2',
                                  np.float64, ('time', 'y', 'x'),
                                  zlib=True)
    var11.standard_name = 'max.curvature2'  # this is a CF standard name
    var11.units = ''
    top_var11 = Pmax2_norm
    var11[iter, :, :] = top_var11

    ncfile.close()


@st.cache
def savebase_netcdf(X, Y, h_base):

    # create netcdf4 file
    ncfilename = ascii_file.replace('.asc', '_base.nc')

    ncfile = netCDF4.Dataset(ncfilename, mode='w', format='NETCDF4')

    x_dim = ncfile.createDimension('x', X.shape[1])
    y_dim = ncfile.createDimension('y', X.shape[0])
    # unlimited axis (can be appended to).
    time_dim = ncfile.createDimension('time', None)

    ncfile.title = ascii_file.replace('.asc', ' analysis')

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

    z[iter, :, :] = h_base

    ncfile.close()


@st.cache
def detect_ridges(X, Y, slope, h, curv_var, curvature_variable, sigma=1.0):

    Pmax1, Pmin1 = surfature(X, Y, h)

    Pmax1_norm = (Pmax1 - np.nanmin(Pmax1)) / \
        (np.nanmax(Pmax1) - np.nanmin(Pmax1))
    Pmin1_norm = - (Pmin1 - np.nanmin(Pmin1)) / \
        (np.nanmax(Pmin1) - np.nanmin(Pmin1))

    H_elems = hessian_matrix(slope, sigma=10.0, order='xy')
    # eigenvalues of hessian matrix
    Pmax2, Pmin2 = hessian_matrix_eigvals(H_elems)

    Pmax2, Pmin2 = surfature(X, Y, slope)


    Pmax2_norm = (Pmax2 - np.nanmin(Pmax2)) / \
        (np.nanmax(Pmax2) - np.nanmin(Pmax2))
    Pmin2_norm = - (Pmin2 - np.nanmin(Pmin2)) / \
        (np.nanmax(Pmin2) - np.nanmin(Pmin2))

    if curv_var == 'elevation':

        Pmax_norm = Pmax1_norm
        Pmin_norm = Pmin1_norm

    elif curv_var == 'slope':

        Pmax_norm = Pmax2_norm
        Pmin_norm = Pmin2_norm

    h_norm = (h - np.nanmin(h)) / (np.nanmax(h) - np.nanmin(h))

    slope_norm = -(slope - np.nanmin(slope)) / \
        (np.nanmax(slope) - np.nanmin(slope))

    print('top variable', curvature_variable)

    if curvature_variable == 'elevation+max.curvature+slope':

        top_var = (h_norm + slope_norm + Pmax_norm) / 3.0

    elif curvature_variable == 'elevation+max.curvature':

        top_var = 0.5 * (h_norm + Pmax_norm)

    elif curvature_variable == 'max.curvature+slope':

        top_var = 0.5 * (slope_norm + Pmax_norm)

    elif curvature_variable == 'elevation+slope':

        top_var = 0.5 * (h_norm + slope_norm)

    elif curvature_variable == 'elevation':

        top_var = h_norm

    elif curvature_variable == 'min.curvature':

        top_var = Pmin_norm

    elif curvature_variable == 'max.curvature':

        top_var = Pmax_norm

    elif curvature_variable == 'slope':

        top_var = slope_norm

    norm_image = cv2.normalize(top_var,
                               None,
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
    print("Loop in {toc - tic:0.4f} seconds")

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

    h_DEM = h

    smoothing_level = st.sidebar.slider("Smoothing", 0, 20, 5)

    # Gaussian blur wants odd value
    smoothing_odd = 2 * smoothing_level + 1
    h = cv2.GaussianBlur(h, (smoothing_odd, smoothing_odd), 0)

    ls = LightSource(azdeg=135, altdeg=45)

    extent = [x_min, x_max, y_min, y_max]

    fig, ax = plt.subplots()

    hill_check = st.sidebar.checkbox('DEM Hillshade')

    if hill_check:

        ax.imshow(ls.hillshade(h, vert_exag=1.0, dx=delta_x, dy=delta_y),
                  cmap='gray',
                  extent=extent,
                  origin='lower')

    else:

        ax.imshow(h, cmap='gray', extent=extent, origin='lower')

    grad_h, h_x, h_y, slope = compute_slope(h, delta_x, delta_y)

    slope = (np.maximum(0.0, np.minimum(slope, 33.0))) / 33.0

    slope_check = st.sidebar.checkbox('Slope')

    slope_opacity = st.sidebar.slider("Slope opacity", 0, 100, 50)
    slope_alpha = slope_opacity / 100.0

    if slope_check:

        ax.imshow(slope,
                  cmap='gray',
                  extent=extent,
                  origin='lower',
                  alpha=slope_alpha)

    st.sidebar.markdown("""---""")

    curv_var = st.sidebar.selectbox('Curvature variable',
                                    ('elevation', 'slope'))

    curvature_variable = st.sidebar.selectbox(
        'Cone top detection variable',
        (
            # 'elevation+min.curvature+slope',
            # 'elevation+min.curvature',
            # 'min.curvature+slope',
            'elevation+max.curvature+slope',
            'elevation+max.curvature',
            'max.curvature+slope',
            'elevation+slope',
            'elevation',
            # 'min.curvature',
            'max.curvature',
            'slope'))

    norm_image = detect_ridges(
        X,
        Y,
        slope,
        h,
        curv_var,
        curvature_variable,
        sigma=10.0,
    )

    det_var_plot_check = st.sidebar.checkbox('Detection variable Plot')

    a_opacity = st.sidebar.slider("Detection variable opacity", 0, 100,
                                  50)
    a_alpha = a_opacity / 100.0

    save_top_var_check = st.sidebar.button('NetCDF for top save')

    if save_top_var_check:

        save_netcdf(ascii_file,
                    X,
                    Y,
                    slope,
                    h_DEM,
                    h,
                    curv_var,
                    curvature_variable,
                    sigma=1.0)

    if det_var_plot_check:

        ax.imshow(norm_image,
                  cmap='gray',
                  extent=extent,
                  origin='lower',
                  alpha=a_alpha)

    st.sidebar.markdown("""---""")

    thresh_check = st.sidebar.checkbox('Threshold')

    thresh_level = st.sidebar.slider("Threshold level", 1, 255, 100)
    thresh_opacity = st.sidebar.slider("Threshold opacity", 0, 100, 50)
    thresh_alpha = thresh_opacity / 100.0

    if thresh_check:

        img = apply_thresh(norm_image)
        img_plot = np.ma.masked_where(img != 255, img)

        # ax.imshow(img_plot, cmap='gray', extent=extent,origin='lower', alpha=thresh_alpha)

        binary = opt_connected_comp(img, h)
        binary_plot = np.ma.masked_where(binary == 1, binary)

        # get a copy of the gray color map
        my_cmap = copy.copy(plt.cm.get_cmap('viridis'))

        ax.imshow(binary,
                  cmap=my_cmap,
                  extent=extent,
                  origin='lower',
                  alpha=thresh_alpha)

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
        print('size', binary.shape[0], binary.shape[1],
              binary.shape[0] * binary.shape[1])
        print('unique', np.unique(skeleton_binary))
        for i in np.unique(skeleton_binary):

            print(i, np.sum(skeleton_binary == i))

        img = cv2.threshold(skeleton_binary, 127, 255,
                            cv2.THRESH_BINARY)[1]  # ensure binary

        img = np.zeros_like(skeleton_binary)
        img[skeleton_binary == 0] = 1
        for i in np.unique(img):

            print('img', i, np.sum(img == i))

        from scipy import ndimage
        labeled, nr_objects = ndimage.label(img > 0.5)
        print("Number of objects is {}".format(nr_objects))
        print('Labels', np.unique(labeled))
        # ax.imshow(labeled, extent=extent,origin='lower', alpha=skeleton_alpha)

        if nr_objects > 1:

            # loop to search for connected components with largest mean elevation
            label_opt = 0
            label_sum = 0
            for label in range(nr_objects + 1):

                label_check = np.sum(labeled == label)
                print('label,label_check', label, label_check)

                if (label_check > label_sum):

                    label_sum = label_check
                    label_opt = label

            img = np.zeros_like(skeleton_binary)
            img[labeled == label_opt] = 1
            # ax.imshow(img, cmap=my_cmap, extent=extent,origin='lower', alpha=skeleton_alpha)

            # remove small holes
            kernel = np.ones((5, 5), np.uint8)

            # The first parameter is the original image,
            # kernel is the matrix with which image is
            # convolved and third parameter is the number
            # of iterations, which will determine how much
            # you want to erode/dilate a given image.
            img_dilation = cv2.dilate(img, kernel, iterations=1)
            img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
            # ax.imshow(img_erosion, cmap=my_cmap, extent=extent,origin='lower', alpha=skeleton_alpha)

            skeleton = 255 * img_erosion
            pruning_size = 0

        else:

            skeleton = 255 * skeleton_binary
            pruning_size = st.sidebar.slider("Skeleton pruning size", 0, 1000,
                                             50)
            pruned_skeleton, segmented_img, segment_objects = pcv.morphology.prune(
                skel_img=skeleton, size=pruning_size)
            skeleton = pruned_skeleton.astype(float)

        if skeleton_alpha > 0:

            ax.imshow(skeleton,
                      cmap=my_cmap,
                      extent=extent,
                      origin='lower',
                      alpha=skeleton_alpha)

        skeleton_vector, closed = raster_to_vector(X, Y, skeleton)

        skeleton_ellipse_check = st.sidebar.checkbox('Skeleton ellipse')
        
        
        skeleton_vector_check = st.sidebar.checkbox('Skeleton vector')
        skeleton_level = st.sidebar.slider("Skeleton vector simplify level", 0,
                                           20, 5)
        skeleton_smoothing_level = st.sidebar.slider(
            "Skeleton vector smoothing level", 0, 20, 0)
        skeleton_range = st.sidebar.slider('Skeleton range of values', 0, 100,
                                           (0, 100))

    else:

        skeleton_ellipse_check = False
        skeleton_vector_check = False

    if skeleton_ellipse_check:
    
        x, y = skeleton_vector.coords.xy
    
        # Fit the contour with an ellipse
        cx, cy, a, b, angle = fitEllipse(x - X[0, 0],
                                         y - Y[0, 0], 2)
                                                     
        # Append the ellipse center coordinates to the lists
        # cx_ellipse.append(float(cx.real))
        # cy_ellipse.append(float(cy.real))

        # Create an ellipse object from the fitting parameters 
        ell = Ellipse((cx.real, cy.real), a.real * 2., b.real * 2.,
                      angle.real)
                                  
        # Get the coordinates of a set of ellipse points
        ell_coord = ell.get_verts()
        x_ellipse = ell_coord[:,0] + X[0, 0]
        y_ellipse = ell_coord[:,1] + Y[0, 0]
                    
        # Plot the points in the absolute coordinate system
        ax.plot(x_ellipse,y_ellipse, 'k-')
                         
        # Plot the center of the ellipse in the absolute coordinate system
        ax.plot(cx.real + X[0, 0], cy.real + Y[0, 0], 'kx')

    if skeleton_vector_check:

        # we compute a vector representation of the skeleton
        # skeleton_vector is a polyline defined by points

        
        
        skeleton_vector, closed = improve_vector(skeleton_vector,closed,skeleton_level,skeleton_smoothing_level)

        print('Skeleton vector',skeleton_vector.geom_type)
        print('Closed', closed)
        ax.plot(*skeleton_vector.xy, 'g')

        if closed:
            # compute buffer area (points within a fixed distance from medial axis)
            path = offset_path(skeleton_vector, dx, False)

            pixel_coordinates = np.c_[X.ravel(), Y.ravel()]

            # find points within path
            skeleton = path.contains_points(pixel_coordinates).reshape(
                X.shape[0], X.shape[1])

        # -------------- FLANK ANALYSIS --------------------

        st.sidebar.markdown("""---""")

        flank_check = st.sidebar.checkbox('Flank')
        flank_var = 'Slow'

        correction_factor = st.sidebar.slider("Scalar dot", 0.00, 1.00, 0.50)

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

        if flank_alpha > 0.0:

            ax.imshow(scal_dot,
                      cmap='gray',
                      extent=extent,
                      origin='lower',
                      alpha=flank_alpha)

        scal_dot_scaled = 255 * scal_dot

        if buffer_check:

            mask_check = True

    else:

        mask_check = False

    if buffer_check:

        # compute buffer area (points within a fixed distance from medial axis)
        path = offset_path(skeleton_vector, buffer_distance, False)
        
        pixel_coordinates = np.c_[X.ravel(), Y.ravel()]

        # find points within path
        img_buffer = path.contains_points(pixel_coordinates).reshape(
            X.shape[0], X.shape[1])

        if buffer_alpha > 0.0:

            ax.imshow(img_buffer,
                      cmap='gray',
                      extent=extent,
                      origin='lower',
                      alpha=buffer_alpha)

        if flank_check:

            mask_check = True

    else:

        mask_check = False

    flank_thr_check = False

    if mask_check:

        # -------------- FLANK THRESHOLD --------------------
        st.sidebar.markdown("""---""")

        flank_thr_check = st.sidebar.checkbox('Mask save')
        flank_radio = st.sidebar.radio('Select flank image:',
                                       ['Buffer', 'Flank', 'Intersection'])

        mask_opacity = st.sidebar.slider("Mask opacity", 0, 100, 50)
        mask_alpha = mask_opacity / 100.0

    if flank_thr_check:

        img_01 = img

        img_2 = img_buffer

        # compute buffer area (points within a fixed distance from medial axis)
        path3 = offset_path(skeleton_vector, 20, False)

        pixel_coordinates = np.c_[X.ravel(), Y.ravel()]

        # find points within path
        img_3 = path3.contains_points(pixel_coordinates).reshape(
            X.shape[0], X.shape[1])

        if flank_radio == 'Buffer':

            mask = img_2

        elif flank_radio == 'Flank':

            mask = np.logical_or(img_01, img_3)

        else:

            # find intersection of buffer and flank
            img = np.logical_and(img_01, img_2)
            img = np.logical_or(img, img_3)
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

        if mask_alpha:

            ax.imshow(mask,
                      cmap='gray',
                      extent=extent,
                      origin='lower',
                      alpha=mask_alpha)

        # Save mask on ascii raster file
        header = "ncols     %s\n" % h.shape[1]
        header += "nrows    %s\n" % h.shape[0]
        header += "xllcenter " + str(np.amin(X)) + "\n"
        header += "yllcenter " + str(np.amin(Y)) + "\n"
        header += "cellsize " + str(np.abs(X[1, 2] - X[1, 1])) + "\n"
        header += "NODATA_value -9999\n"

        output_full = ascii_file.replace('.asc', '_mask.asc')

        print('mask min max', np.nanmin(img), np.nanmax(img))

        edt = ndimage.distance_transform_edt(skeleton == 0)

        edt *= dx

        dist_mask = edt
        dist_mask[mask == 0] = -9999

        np.savetxt(temp_path + output_full,
                   np.flipud(dist_mask),
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
        min_base_distance = st.sidebar.slider("Minimum base distance", 10,
                                              buffer_distance, 10)

    else:

        volume_check = False

    if volume_check:

        fig_c, ax_c = plt.subplots()

        cn = ax_c.contour(X, Y, mask, 1)

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

                xv = [vv[0][0] for vv in pp.iter_segments()]
                yv = [vv[0][1] for vv in pp.iter_segments()]

                if len(xv) > base_len:

                    x_cnt = xv
                    y_cnt = yv
                    base_len = len(xv)

        line_cnt = LineString(zip(x_cnt, y_cnt))
        line_cnt = line_cnt.simplify(20.0, preserve_topology=False)

        x_cnt, y_cnt = line_cnt.coords.xy

        coords_cnt = []
        counter = 0
        for (x, y) in zip(x_cnt, y_cnt):

            dist = skeleton_vector.distance(Point(x, y))

            if (dist > min_base_distance):

                # coords_cnt.append((x,y))

                coords_cnt.insert(counter, (x, y))
                counter += 1
                ax.plot(x, y, 'xk')

            else:

                counter = 0

        line_cnt = LineString(coords_cnt)

        print('line_cnt length', line_cnt.length)

        C_base, h_base = plane_fit(line_cnt, X, Y, h)

        save_base_var_check = st.sidebar.button('NetCDF for base save')

        if save_base_var_check:

            savebase_netcdf(X, Y, h_base)

        h_mask = np.ma.masked_where(mask == 0, h - h_base)

        flank_vol = dx * dy * np.nansum(h_mask)

        print('flank_vol', flank_vol)

        st.sidebar.write('Flank volume =', flank_vol, 'm3')
        st.sidebar.write('Base plane z =', C_base[2], '+', C_base[0],
                         'x_rel +', C_base[1], 'y_rel')

    # -------------- SLOPE ANALYSIS --------------------

    if flank_thr_check:

        st.sidebar.markdown("""---""")

        slope_check = st.sidebar.checkbox('Slope analysis')
        slope_smooth_level = st.sidebar.slider("Slope smoothing level", 0, 20,
                                               5)
        slope_fit_check = st.sidebar.checkbox('Slope fitting')

    else:

        slope_check = False

    if slope_check:

        # aspect from topography
        h_x_filtered = filters.gaussian(h_x, slope_smooth_level)
        h_y_filtered = filters.gaussian(h_y, slope_smooth_level)

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

        if slope_fit_check:

            p0 = (33, 0.01, 0.0)  # start with values near those we expect
            params, cv = scipy.optimize.curve_fit(monoExp, dist_half[1:],
                                                  slp_mean[1:], p0)
            m_fit, t_fit, b_fit = params

            ax_slope2.plot(dist_half,
                           monoExp(dist_half, m_fit, t_fit, b_fit),
                           '--',
                           label="fitted")

    # -------------- SYNTHETIC CONE --------------------

    if volume_check:

        st.sidebar.markdown("""---""")

        synth_check = st.sidebar.checkbox('Synthetic cone')
        if skeleton_ellipse_check:
        
            synth_ellipse_check = st.sidebar.checkbox('Elliptic synthetic cone top')

        top_linear_check = st.sidebar.checkbox('Linear fit of cone top')
        
            
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
    
        

        if top_linear_check:

            C_top, h_top = plane_fit(skeleton_vector, X, Y, h)

        else:

            h_top = h



        # compute small buffer area (points within distance 1.0 from top)        
        if synth_ellipse_check:
        
            coords = np.array([[x, y] for x, y in zip(x_ellipse, y_ellipse)])
            path = offset_path(LinearRing(coords), 1.0, False)

        else:
            
            path = offset_path(skeleton_vector, 1.0, False)
        

        pixel_coordinates = np.c_[X.ravel(), Y.ravel()]

        # find points within path
        img_buffer = path.contains_points(pixel_coordinates).reshape(
            X.shape[0], X.shape[1])

        edt, inds = ndimage.distance_transform_edt(img_buffer == 0,
                                                   return_indices=True)

        # edt, inds = ndimage.distance_transform_edt(skeleton == 0,
        #                                            return_indices=True)

        edt *= dx

        # width_cone = h_cone / np.tan(np.radians(cr_slope))
        # edt = np.minimum(edt, width_cone)

        # synth_cone = -(edt - width_cone) / width_cone * h_cone

        # print('Volume = ',np.sum(synth_cone))

        h_temp = h_top[tuple(inds)] - edt * np.tan(np.radians(cr_slope))
        smoothing = 39
        h_temp = cv2.GaussianBlur(h_temp, (smoothing, smoothing), 0)

        h0 = 0.0

        synth_cone = np.maximum(h_base, h_temp + h0)
        synth_cone = np.ma.masked_where(mask == 0, synth_cone)
        synth_vol0 = dx * dy * np.nansum(synth_cone - h_base)
        print('synth_vol0', synth_vol0)

        if synth_vol0 < flank_vol:

            h2 = 300.0
            synth_cone = np.maximum(h_base, h_temp + h2)
            synth_cone = np.ma.masked_where(mask == 0, synth_cone)
            synth_vol2 = dx * dy * np.nansum(synth_cone - h_base)
            print('synth_vol2', synth_vol2)

        for i in range(20):

            h1 = 0.5 * (h0 + h2)
            synth_cone = np.maximum(h_base, h_temp + h1)
            synth_cone = np.ma.masked_where(mask == 0, synth_cone)
            synth_vol1 = dx * dy * np.nansum(synth_cone - h_base)
            print('h1', h1)
            print('synth_vol1', synth_vol1)

            if (synth_vol1 < flank_vol):

                h0 = h1

            else:

                h2 = h1

        synth_cone = np.maximum(h_base, h_temp + h1)

        # synth_cone = np.maximum(h_base,h_base + h_cone - edt * np.tan(np.radians(cr_slope)))

        ax.imshow(ls.hillshade(synth_cone,
                               vert_exag=1.0,
                               dx=delta_x,
                               dy=delta_y),
                  cmap='gray',
                  extent=extent,
                  origin='lower',
                  alpha=synth_alpha)

        # synth_mask = np.ma.masked_where(mask > 0, synth_cone)

        # synth_vol = dx * dy * np.nansum(synth_cone-h_base)

        # print('synth_vol', synth_vol)

        st.sidebar.write('Synthetic cone flank volume =', synth_vol1, 'm3')

        df = pd.DataFrame({
            'scii_file': [ascii_file],
            'smoothing': [smoothing_level],
            'curvature variable': [curvature_variable],
            'threshold level': [thresh_level],
            'skeleton pruning size': [pruning_size],
            'skeleton vector simplify level': [skeleton_level],
            'skeleton vector smoothing level': [skeleton_smoothing_level],
            'skeleton range of values': [skeleton_range],
            'scalar dot': [correction_factor],
            'buffer_distance': [buffer_distance],
            'flank image': [flank_radio],
            'minimum base distance': [min_base_distance],
            'slope smoothing level': [slope_smooth_level],
            'linear fit top check': [top_linear_check],
            'slope angle': [cr_slope]
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
                   np.flipud(h_synth),
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
