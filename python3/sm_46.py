# Plot the symmetry map of 4-fold and 4-fold 
# Reference: Kevin Whitham "Structure and electronic properties of Lead-Selenide Nanocrystal Solids." Doctoral dissertation (2016) 
# Reference: Mickel, Walter, et al. "Shortcomings of the bond orientational order parameters for the analysis of disordered particulate matter." The Journal of chemical physics (2013)
# 20190602 Jen-Yu Huang, Cornell University
# License: GNU Public License (GPL) v.3.0

# general math
import numpy as np
from scipy import stats # for stats.mode

# for data structures
import collections
import scipy.sparse as sparse

# for fitting the radius distribution
from scipy.optimize import curve_fit
from scipy.stats import linregress

# for disabling annoying warnings
import warnings

# plotting
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Circle, Arrow
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable, Size

# Voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d

# for finding particle centers, diameters, etc.
from skimage.measure import regionprops
from skimage.filters import threshold_otsu,threshold_local, threshold_isodata
from skimage.morphology import watershed, remove_small_objects, binary_erosion, binary_dilation, binary_closing, binary_opening
from skimage.feature import peak_local_max
from scipy import ndimage
from skimage.draw import circle
from skimage.morphology import disk
from numpy import genfromtxt

# for command line interface
import argparse
from skimage import io as skimio
from os import path
from os import walk

# regular expressions for parsing scalebar filenames
import re

# for matching the scale bar
from skimage.feature import match_template

# minkowski structure metric function written and compiled with Cython
from minkowski_metric import minkowski
#import pyximport; pyximport.install()
#import minkowski_metric




def make_binary_image(im, white_background, min_feature_size, small):

    image = im.astype('float32')

    if white_background:
        image = np.abs(image-np.max(image))

    # get rid of large background patches before local thresholding
    # do a global threshold
    thresh = threshold_isodata(image)
    binary = image > thresh
    
    # get rid of speckle
    # this is not good for very small particles
    if not small:
        binary_closing(binary, selem=disk(min_feature_size), out=binary)

    # make a map of the distance to a particle to find large background patches
    # this is a distance map of the inverse binary image
    distance = ndimage.distance_transform_edt(1-binary)
    
    # dilate the distance map to expand small voids
    #distance = ndimage.grey_dilation(distance,size=2*min_feature_size)
    
    # do a global threshold on the distance map to select the biggest objects
    # larger than a minimum size to prevent masking of particles in images with no empty patches
    dist_thresh = threshold_isodata(distance)
    mask = distance < dist_thresh  
    
    # remove areas of background smaller than a certain size in the mask
    # this fills in small pieces of mask between particles where the voronoi
    # vertices will end up
    # this gets rid of junk in the gap areas
    binary_opening(mask, selem=disk(min_feature_size), out=mask)
    
    binary = binary * mask

    # get rid of speckle
    binary = remove_small_objects(binary,min_size=max(min_feature_size,2))
    
    # 3 iterations is better for large particles with low contrast
    binary = ndimage.binary_closing(binary,iterations=3)

    

    return binary, mask
    
def adaptive_binary_image(im, white_background, min_feature_size, std_dev, mask):

    image = im.astype('float32')

    if white_background:
        image = np.abs(image-np.max(image))
    
    # the block size should be large enough to include 4 to 9 particles
    local_size = 40*min_feature_size
    binary = image > threshold_local(image,block_size=local_size+1)
    
    # plt.imshow(binary)
    # plt.show()
  
    # close any small holes in the particles
    # 3 iterations is better for large particles with low contrast
    binary = ndimage.binary_closing(binary,iterations=1)
    # binary_closing(binary, selem=disk(int(max((0.414*(min_feature_size-std_dev)),2)/2.0)), out=binary)
    
    # plt.imshow(binary)
    # plt.show()

    # remove speckle from background areas in the binary image
    binary = binary * mask #binary_erosion(mask, selem=disk(int(min_feature_size)))
    binary_opening(binary, selem=disk(int(max((min_feature_size-3.0*std_dev),2)/2.0)), out=binary)

    # make a distance map of the inverted image
    distance = ndimage.distance_transform_edt((1-binary))

    # do a global threshold on the distance map to select the biggest objects
    # larger than a minimum size to prevent masking of particles in images with no empty patches
    dist_thresh = threshold_isodata(distance)
    new_mask = distance < dist_thresh

    # thresholding is selecting too much background, try AND'ing it with the binary image
    # this might only be necessary for small particles
    new_mask = new_mask * binary
    
    # remove areas of background in the mask smaller than a certain size
    # this fills in small pieces of mask between particles where the voronoi
    # vertices will end up
    # min_feature_size here should be the radius found by global threshold, so triple it
    # to close any particle sized holes
    dilation_size = max(int(1),int(3 * min_feature_size))
    new_mask = binary_closing(new_mask, selem=np.ones((dilation_size,dilation_size)))
    
    plt.imshow(new_mask)
    plt.show()
    
    return binary, new_mask

def morphological_threshold(im, white_background, mean_radius, min_feature_size, small, mask):

   
    im_mod = np.array(im, dtype='float64')

    if white_background:
        im_mod = np.abs(im_mod-np.max(im_mod))
        
    # subtract the mean before running match_template
    # not sure this works quite right
    im_mod = im_mod - np.mean(im_mod)

    # set large areas of background to zero using the mask
    #im_mod = im_mod * mask
    
    

    if small:
        template_matrix = np.pad(disk(max(2,int(mean_radius))),pad_width=max(2,int(mean_radius)), mode='constant',constant_values=0)
    else:
        template_matrix = np.pad(disk(int(mean_radius/4)),pad_width=int(mean_radius), mode='constant',constant_values=0)

    matched_im = match_template(im_mod,template=template_matrix,pad_input=True)
    
  

    thresh = threshold_isodata(matched_im)
    matched_im_bin = matched_im > thresh
    
    matched_im_bin *= mask

    distance = ndimage.distance_transform_edt(matched_im_bin)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=int(min_feature_size))

    local_maxi = peak_local_max(distance,indices=False,min_distance=int(min_feature_size))
    markers = ndimage.label(local_maxi)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels_th = watershed(-distance, markers, mask=matched_im_bin)

   

    return labels_th, matched_im_bin


def get_particle_centers(im, white_background, pixels_per_nm, morph, small):

    if pixels_per_nm == 0:
        # default value
        pixels_per_nm = 1.0

    # minimum size object to look for
    min_feature_size = int(3) #int(3*pixels_per_nm)

    global_binary, mask = make_binary_image(im, white_background, min_feature_size, small)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(global_binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=min_feature_size)

    # min_distance=5 for large particles
    local_maxi = peak_local_max(distance,indices=False,min_distance=min_feature_size)
    markers = ndimage.label(local_maxi)[0]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        global_labels = watershed(-distance, markers, connectivity=None, offset=None, mask=global_binary)

    # get the particle radii
    global_regions = regionprops(global_labels, coordinates= 'xy')
    gobal_radii = []
    
    for props in global_regions:
        # define the radius as half the average of the major and minor diameters
        gobal_radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # minimum size object to look for
    global_mean_radius = np.mean(gobal_radii)*pixels_per_nm
    global_radii_sd = np.std(gobal_radii)*pixels_per_nm
    
    print('Mean radius global threshold (px): %(rad).2f SD: %(sd).2g' % {'rad':global_mean_radius, 'sd':global_radii_sd})
    
    feature_size = int(max(global_mean_radius, min_feature_size))
    std_dev = int(max(global_radii_sd, min_feature_size))
    
    adaptive_binary, adaptive_mask = adaptive_binary_image(im, white_background, feature_size, std_dev, mask)

    # create a distance map to find the particle centers
    # as the points with maximal distance to the background
    distance = ndimage.distance_transform_edt(adaptive_binary)

    # dilate the distance map to merge close peaks (merges multiple peaks in one particle)
    distance = ndimage.grey_dilation(distance,size=feature_size)

    local_maxi = peak_local_max(distance,indices=False,min_distance=feature_size)
    markers = ndimage.label(local_maxi)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        adaptive_labels = watershed(-distance, markers, connectivity=None, offset=None, mask=adaptive_binary)

    adaptive_regions = regionprops(adaptive_labels, coordinates = 'xy')
    adaptive_radii = []
    
    for props in adaptive_regions:
        # define the radius as half the average of the major and minor diameters
        adaptive_radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    # minimum size object to look for
    adaptive_radii_sd = np.std(adaptive_radii)
    adaptive_mean_radius = np.mean(adaptive_radii)
    print('Mean radius adaptive threshold (px): %(rad).2f SD: %(sd).2g' % {'rad':adaptive_mean_radius*pixels_per_nm, 'sd':adaptive_radii_sd*pixels_per_nm})
    
    
    # morphological thresholding
    if morph:
        print('Using morphological threshold')
        labels, morph_binary = morphological_threshold(im, white_background, int(adaptive_mean_radius*pixels_per_nm), int(adaptive_mean_radius*pixels_per_nm)/2, small, adaptive_mask)
        regions = regionprops(labels, coordinates='xy')
        binary = morph_binary
    elif global_radii_sd/global_mean_radius < adaptive_radii_sd/adaptive_mean_radius:
        print('Using global threshold')
        regions = global_regions
        binary = global_binary
        labels = global_labels
    else:
        print('Using adaptive threshold')
        regions = adaptive_regions
        binary = adaptive_binary
        labels = adaptive_labels

  

    # get the particle centroids again, this time with better thresholding
    pts = []
    radii = []

    for props in regions:
        # centroid is [row, col] we want [col, row] aka [X,Y]
        # so reverse the order
        pts.append(props.centroid[::-1])

        # define the radius as half the average of the major and minor diameters
        radii.append(((props.minor_axis_length+props.major_axis_length)/4)/pixels_per_nm)

    if morph:
        print('Mean radius morphological threshold (px): %(rad).2f SD: %(sd).2g' % {'rad':np.mean(radii)*pixels_per_nm, 'sd':np.std(radii)*pixels_per_nm})
        
        
    return np.asarray(pts,dtype=np.double), np.asarray(radii,dtype=np.double).reshape((-1,1)), adaptive_mask, binary

def get_image_scale(im):

    scale = 0.0
    topleft = (0,0)
    bottomright = (0,0)

    input_path = path.normpath('../resources/scalebars')

    # load all files in scalebars directory and sub-directories
    scalebar_filesnames = []
    for (dirpath, dirnames, filesnames) in walk(input_path):
        scalebar_filesnames.extend(filesnames)

    scale_bars = []
    for filename in scalebar_filesnames:
        ipart = 1 #first number in the name
        fpart = 0.0 #second number in the name

        reg_result = re.search('Scale_(\d*)p*(\d*)',filename)

        if reg_result:

            if reg_result.group(1):
                ipart = int(reg_result.group(1))

            if reg_result.group(2):
                fpart = float('0.'+reg_result.group(2))

            px_per_nm = ipart+fpart

            # images of scale bars to match with the input image
            # second element is the scale in units of pixels/nm
            scale_bars.append([skimio.imread(input_path+'/'+filename,as_grey=True,plugin='matplotlib'),px_per_nm])

    match_score = []
    # Match the database scale bar with the image read
    for scale_bar,pixels_per_nm in scale_bars:

        result = match_template(im,template=scale_bar)

        ij = np.unravel_index(np.argmax(result), result.shape)
        row, col = ij

        match_score.append(result[row][col])

        # the match score should be about 0.999999
        if result[row][col] > 0.99:

            topleft = (row,col)
            bottomright = (row+scale_bar.shape[0],col+scale_bar.shape[1])

            scale = pixels_per_nm

            print('Scale: '+str(scale)+' pixels/nm, Score: '+str(np.max(match_score)))
            break # we found it, stop looking

    if np.max(match_score) < 0.99:
        print('!!!!!!!!!!!!!!!!!!!! No scale bar found !!!!!!!!!!!!!!!!!!!!!')

    return [np.double(scale),(topleft,bottomright)]

def create_custom_colormap():

    # get a color map for mapping metric values to colors of some color scale
    value_rgb_pairs = []
    rgb_array = np.asarray([[0,0,0],[255,0,0],[255,50,34],[255,109,59],[255,177,102],[255,220,125],[255,245,160],[255,245,192],[255,255,255],[212,251,255],[160,253,255],[120,226,255],[81,177,255],[55,127,255],[31,81,255],[0,13,255]],dtype='f4')
    rgb_array /= 255
    rgb_list_norm = []

    for value, color in zip(np.linspace(0,1,16),rgb_array):
        value_rgb_pairs.append((value,color))

    return LinearSegmentedColormap.from_list(name="custom", colors=value_rgb_pairs, N=16)
def plot_symmetry(im, msm, vor, radii,bond_order, mlf, symmetry_colormap, mask, no_fill, map_edge_particles,area):

    cell_patches = []
    metric_list = []
    orientation_angle_list = []
    i=0
    count_4, count_6 = 0.0,0.0

    for region_index,metric,orientation_angle in msm:
        plot_this_cell = 0
        region_index = int(region_index)
        region = vor.regions[region_index]
        verts = np.asarray([vor.vertices[index] for index in region])

        x_center = int(round(np.mean(verts[:,1])))
        y_center = int(round(np.mean(verts[:,0])))


        # don't plot cells inside masked off regions of the image (blank patches)
        int_verts = np.asarray(verts,dtype='i4')
        if map_edge_particles:
            if np.any(mask[int_verts[:,1],int_verts[:,0]] == 1):
                plot_this_cell = 1
        if area:    
            if np.all(mask[int_verts[:,1],int_verts[:,0]] > 0):
                temp_cen = np.asarray([np.sqrt(np.abs(b-x_center)**2 + np.abs(a-y_center)**2) for a,b in int_verts])
                if np.mean(temp_cen) < np.mean(radii)*pixels_per_nm*1.6:
                    # print('out')
                    # print x_center,y_center
                    plot_this_cell = 0 
                else:
                    plot_this_cell = 1
        else:
            if np.all(mask[int_verts[:,1],int_verts[:,0]] > 0):
                plot_this_cell = 1


        if plot_this_cell:
            if no_fill:
                cell_patches.append(Polygon(verts,closed=True,facecolor='none',edgecolor='r'))
            else:
                cell_patches.append(Polygon(verts,closed=True,edgecolor='none'))
            metric_list.append(mlf[i])
        i=i+1    

                    
    if no_fill:
        pc = PatchCollection(cell_patches,match_original=True,alpha=0.4)
    else:
        pc = PatchCollection(cell_patches,match_original=False, cmap=symmetry_colormap, edgecolor='k', alpha=1)
        pc.set_array(np.asarray(metric_list))

    # for e in metric_list:
    #     if e> 0.5:
    #         count_4 +=1
    #     if e<-0.5:
    #         count_6 +=1
    # ratio_46 = round(count_4/(count_4+count_6),3)
    # ratio_4t = round(count_4/(len(metric_list)),3)
    # ratio_6t = round(count_6/(len(metric_list)),3)
    # print("symmetry4:",count_4,"symmetry6:", count_6,"ratio_46:", ratio_46,"ratio_4total:", ratio_4t,"ratio_6total:", ratio_6t)


        
    plt.gca().add_collection(pc)

    # set the limits for the plot
    # set the x axis range
    plt.gca().set_xlim(0, im.shape[1])

    # set the y-axis range and flip the y-axis
    plt.gca().set_ylim(im.shape[0], 0)

    # save this plot to a file
    plt.gca().set_axis_off()
  
    
    if not no_fill:
        # add the colorbar
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes(position='right', size='5%', pad = 0.05)
        cbar = plt.colorbar(pc, cax=cax)
        cax.set_xlabel('$\Psi_'+'4'+'$', fontsize=18)
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_label_coords(0.5, 1.04)

        # cbar.set_label('$\Psi_'+str(6)+'$',rotation=270)
    
    # plt.savefig(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_map.png',bbox_inches='tight',dpi=300)

    # plot a histogram of the Minkowski structure metrics
    # plt.figure(2)
    # plt.hist(metric_list,bins=len(msm)/4)
    # plt.xlabel('$\Psi_'+str(bond_order)+'$')
    # plt.ylabel('Count')
    # plt.savefig(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_hist.png', bbox_inches='tight')
    

def plot_particle_outlines(im, pts, radii, pixels_per_nm):
    
    cell_patches = []
    for center,radius in zip(pts,radii):
        cell_patches.append(Circle(center,radius*pixels_per_nm,facecolor='none',edgecolor='r'))
            
    pc = PatchCollection(cell_patches,match_original=True,alpha=1)
        
    plt.gca().add_collection(pc)

    # set the limits for the plot
    # set the x axis range
    plt.gca().set_xlim(0, im.shape[1])

    # set the y-axis range and flip the y-axis
    plt.gca().set_ylim(im.shape[0], 0)

    # save this plot to a file
    plt.gca().set_axis_off()
    
    # plt.savefig(output_data_path+'/'+filename+'_particle_map.png',bbox_inches='tight',dpi=300)

#  Main code

   
im = skimio.imread('PATH of FILE',as_grey=True,plugin='matplotlib')
im_original = np.empty_like(im)
np.copyto(im_original,im)

# output_data_path = path.dirname('')
# filename = str.split(path.basename(''),'.')[0]

background = 1

pixels_per_nm,bar_corners = get_image_scale(im)
if background:
    im[bar_corners[0][0]:bar_corners[1][0]+1,bar_corners[0][1]:bar_corners[1][1]+1] = np.max(im)
else:
    im[bar_corners[0][0]:bar_corners[1][0]+1,bar_corners[0][1]:bar_corners[1][1]+1] = 0

pts, radii, mask, binary = get_particle_centers(im,background,pixels_per_nm,1,0) #mor,smnall
assert len(pts) == len(radii)


vor = Voronoi(points=pts)
# voronoi_plot_2d(vor)
# plt.show()

metric_list4= []
metric_list6= []
metric_list_final = []
msm = minkowski(vor.vertices,vor.regions,4,(im.shape[1],im.shape[0]))
msm6 = minkowski(vor.vertices,vor.regions,6,(im.shape[1],im.shape[0]))

for a,b,c in msm:
    metric_list4.append(b)
for x,y,z in msm6:
    metric_list6.append(y)

for i in range(len(metric_list4)):
    if metric_list4[i] >= metric_list6[i]:
        metric_list_final.append(metric_list4[i])
    if metric_list4[i] < metric_list6[i]:
        metric_list_final.append(-metric_list6[i])



#plot outline

# plt.figure(1)
# plt.subplot(111)
# implot = plt.imshow(im_original)
# implot.set_cmap('gray')
# plot_particle_outlines(im, pts, radii, pixels_per_nm)
# plt.show()

#plot binary

# plt.figure(1)
# plt.subplot(111)
# implot = plt.imshow(mask)
# plt.show()


plt.figure()
plt.clf()
plt.subplot(111)
implot = plt.imshow(im_original)
implot.set_cmap('gray')
symmetry_colormap = plt.get_cmap('RdBu_r')
plot_symmetry(im, msm, vor,radii, 4, metric_list_final, symmetry_colormap, mask,0, 0,1) #no fill, edge, areafilter(need to set the value)
plt.show()



# plt.savefig(output_data_path+'/'+filename+'_Psi'+str(bond_order)+'_map.png',bbox_inches='tight',dpi=300)
    
