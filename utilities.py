from operator import itemgetter
import os
import xml.etree.ElementTree as ET

# find all videos whith a certain directory 
def find_videos(path, extension):
    if not extension:
        extension="MOV"
    videos_names = []
    files_path = []
    for file in os.listdir(path):
        if file.endswith(f".{extension}"):
                videos_names.append(file)
                files_path.append(os.path.join(path,file))
    return(files_path, videos_names)


# check if folder exists, and create one if it doesn't 
def create_folder_for_video (video_path, extention):
    if not os.path.exists(video_path+extention):
        os.mkdir(video_path+extention)

# extract parameters from the xml camera calibration file 
def get_parameters (video_path):
    parameters_dict = {}
    if os.path.exists(f"{video_path[:-4]}_camera_calibration.xml"):
        xmlTree = ET.parse(f"{video_path[:-4]}_camera_calibration.xml")
        for elem in xmlTree.iter():
            parameters_dict[elem.tag] = elem.text
    else:
        parameters_dict['k1'] = 0 
        parameters_dict['k2'] = 0
        parameters_dict['p1'] = 0
        parameters_dict['p2'] = 0
        parameters_dict['f'] = 0

    return parameters_dict
 

