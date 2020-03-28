from operator import itemgetter
import os
import xml.etree.ElementTree as ET

# find all videos whith a certain directory 
def find_videos(path):
    videos_names = []
    files_path = []
    for file in os.listdir(path):
        if file.endswith(".MOV"):
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
    xmlTree = ET.parse(video_path[:-4]+'_camera_calibration.xml')
    for elem in xmlTree.iter():
        parameters_dict[elem.tag] = elem.text
    return parameters_dict

