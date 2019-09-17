import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.path as mpath
from PIL import Image
import numpy as np
import os
import cv2

################
## FUNCTIONS ##
################
def read_json_annotated_images(json_path):
    """
    The annotation are supposed to match the VGG Image Annotator format (json file)
    The json file is composed of
        - "_via_settings" : annotation editor settings and annotation project name
        - "_via_img_metadata" : lists the annotated image, for each image provides
            the path/url, the regions of interest in the image, the shape of each
            region, the annotations corresponding to each region, the annotation
            associated to the whole image.
        - "_via_attributes": details all region and image attributes/annotations
    :param json_path: path to the json file that describes the annotations of a set of images
    :return: a dictionary containing the path to the image and associated metadata
    """
    a_full  = json.load(open(json_path))
    metadata = []
    for im in list((a_full["_via_img_metadata"]).keys()):
        dict = a_full["_via_img_metadata"][im]
        dict.pop("size")
        metadata.append(dict)
    avail_regions = a_full["_via_attributes"]["region"]["Name"]["options"]
    return metadata, list(avail_regions.keys())


def __patch_rectangle(dict_shape):
    x = dict_shape["x"]
    y = dict_shape["y"]
    w = dict_shape["width"]
    h = dict_shape["height"]
    vertice = np.array([[x,x,x+w,x+w],[y,y+h,y,y+h]])
    vertice = vertice.transpose()
    return mpatch.Rectangle((x,y), w, h), vertice


def __patch_polygon(dict_shape):
    vertice = np.asarray([dict_shape["all_points_x"],dict_shape["all_points_y"]])
    vertice = vertice.transpose()
    return mpatch.Polygon(vertice, closed=True), vertice


def show_annotated_image(ax, metadata, list_regions, list_color_regions):
    """
    Plot the image and the annotated regions defined in the image on provided AX
    :param AX: axes instance
    :param METADATA: dictionary containing the path to the image and associated metadata
    :param LIST_REGIONS: list of the available region names/types
    :param LIST_COLOR_REGIONS: list of region colors, same length as LIST_REGIONS
    :return: updated AX
    """
    # check inputs
    assert len(list_regions)==len(list_color_regions)
    # Available region shapes
    switcher_shape = {"rect": __patch_rectangle, \
                      "polygon": __patch_polygon}
    # Available region types
    swicther_color = {}
    reg_j = 0
    for reg_type in list_regions:
        swicther_color[reg_type]=list_color_regions[reg_j]
        reg_j+=1
    im_array = np.asarray(Image.open((metadata["filename"]).strip()))
    ax.imshow(im_array)
    for reg_dict in metadata["regions"]:
        reg_c = swicther_color.get(reg_dict["region_attributes"]["Name"])
        func_patch = switcher_shape.get(reg_dict["shape_attributes"]["name"])
        patch, dummy = func_patch(reg_dict["shape_attributes"])
        patch.set_fill(False)
        patch.set_edgecolor(reg_c)
        patch.set_linewidth(1)
        ax.add_patch(patch)
    # color legend
    (xmin, xmax) = ax.get_xlim()
    (ymin, ymax) = ax.get_ylim()
    print(list_regions)
    for reg_j in range(len(list_regions)):
        ax.text(xmin+(xmax-xmin)/20., ymax-(reg_j+1)*(ymax-ymin)/20., list_regions[reg_j], \
                horizontalalignment='left', verticalalignment='top', \
                color=list_color_regions[reg_j])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels('',visible=False)
    ax.set_yticklabels('',visible=False)
    return ax


def convert_contour_into_mask(metadata, region_name, region_shape, ax=None):
    """
    Convert polygon and rectangular shape definde with VIA application into mask
    :param metadata: dictionary containing the path to the image and associated metadata
    :param region_name: string, the type (class) of the region(s) of interest
    :param region_shape: string, the shape of contour we are interested in (rect, polygon...)
    :param ax: axes instance if plot is needed
    :return: the mask (and updated axes instance)
    """
    # Available region shapes
    switcher_shape = {"rect": __patch_rectangle, \
                      "polygon": __patch_polygon}
    im_array = np.asarray(Image.open((metadata["filename"]).strip()))
    w = im_array.shape[1]
    h = im_array.shape[0]
    y,x = np.mgrid[:h,:w]
    points = np.transpose((x.ravel(),y.ravel()))
    mask = np.zeros(points.shape[0])
    for reg_dict in metadata["regions"]:
        if ((reg_dict["region_attributes"]["Name"]==region_name) &
            (reg_dict["shape_attributes"]["name"]==region_shape)):
            func_patch = switcher_shape.get(reg_dict["shape_attributes"]["name"])
            patch, vert_xy = func_patch(reg_dict["shape_attributes"])
            path = mpath.Path(vert_xy)
            mask += path.contains_points(points)
    mask = mask>0
    mask = mask.reshape((im_array.shape[0],im_array.shape[1]))
    if ax is not None:
        ax.imshow(mask)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels('',visible=False)
        ax.set_yticklabels('',visible=False)
        return mask, ax
    else:
        return mask


def convolve2d(image,kernel):
    """
    MxN 2D array convolved by mxn 2D array results in (M-m+1)x(N-n+1) array with
    valid convolution results.
    kernel => 1D array (m-1)x(N+n)+n
    im => 1D (M+m)x(N+n)
    ==> Mx(N+n)-((m-1)x(N+n)+n)+1 = (N+n)x(M-m+1)-(n-1)
    :param im: MxN array
    :param kernel: mxn array
    :return:
    """
    Mim, Nim = image.shape
    mk, nk = kernel.shape
    # Convert 2D kernel into 1D vector
    kernel1d = np.concatenate([kernel,np.zeros((mk,Nim+2))],axis=1)
    kernel1d = kernel1d.ravel()
    kernel1d = kernel1d[0:((mk-1)*(nk+Nim+2)+nk)] # (m-1)*N+n
    # Pading image with 0
    im1d = np.zeros((Mim+2,Nim+2))
    im1d[1:1+Mim,1:1+Nim] = image
    # Convert image into 1D vector
    im1d = np.concatenate([im1d,np.zeros((Mim+2,nk))],axis=1)
    im1d = im1d.ravel() # (M+2)*(N+2+n) size
    # Convolve
    conv = np.convolve(im1d,kernel1d,mode='same')
    conv = conv.reshape((Mim+2,nk+Nim+2))
    conv = conv[1:1+Mim,1:1+Nim]
    return conv


def __convert_vector_into_integer_list(vector):
    out_list = []
    for j in range(vector.size):
        out_list.append(int(vector[j]))
    return out_list


def convert_mask_into_polygon(mask_path):
    """
    :param mask_path: path to amask image (tif or png format)
    :return: the coordinates of vertices of the polygon (// VIA app)
    """
    # Finding disconnected regions
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(mask, 5,255,cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours[0]:
        vertice = np.array(cnt)
        vertice = np.squeeze(vertice)
        all_x = __convert_vector_into_integer_list(vertice[:,0])
        all_y = __convert_vector_into_integer_list(vertice[:,1])
        polygons.append((all_x,all_y))
    return polygons


def add_masks_to_metadata(json_path, masks_dir, json_out="out.json", region_name=""):
    # Open json file
    metadata = json.load(open(json_path))
    # List images with available mask
    dir_list = os.listdir(masks_dir)
    masks_list = [im_name for im_name in dir_list if "mask" in im_name]
    images_list = [(im_name.split('_mask'))[0] for im_name in masks_list]
    key_list = list(metadata["_via_img_metadata"].keys())
    for key in key_list:
        image_name = (((metadata["_via_img_metadata"][key]["filename"].split('/'))[-1]).split('.'))[0]
        if image_name in images_list:
            mask_path = masks_dir+masks_list[images_list.index(image_name)]
            contours = convert_mask_into_polygon(mask_path)
            for cnt in contours:
                new_region = {"shape_attributes": {"name":"polygon","all_points_x":cnt[0],
                                                   "all_points_y":cnt[1]},
                              "region_attributes":{"Name":region_name}}
                metadata["_via_img_metadata"][key]["regions"].append(new_region)
    f_out = open(json_out,'w')
    json.dump(metadata,f_out,indent=3)
    f_out.close()
    return metadata


################
##    MAIN    ##
################
main_dir = "/home/lorraine/PycharmProjects/ML_crack_classifier/"
data_dir = main_dir+"/data/Concrete_pylon_crack_annotated/"
# Add mask to the json file that contains annotations
# metadata, regions = read_json_annotated_images(main_dir+"via_project_fissures.json")
# out_metadata = main_dir+"new_via_project_fissures.json"
# metadata = add_masks_to_metadata(main_dir+"via_project_fissures.json",
#                                  main_dir+"data/TIF/", json_out=out_metadata,
#                                  region_name="crack")
metadata, regions = read_json_annotated_images(data_dir+"concrete_pylon_crack_annot_via.json")
colors = ["r","b","y","g","m","w"]


# show annotated images
image_num = 80
fig = plt.figure(0,figsize=(12,8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1 = show_annotated_image(ax1, metadata[image_num], regions, colors[0:len(regions)])
ax2 = convert_contour_into_mask(metadata[image_num], 'crack', 'polygon', ax=ax2)
plt.show()



