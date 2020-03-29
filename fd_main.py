import argparse
import cv2 
import numpy as np
from operator import itemgetter
import os
import pandas as pd
import time

from utilities import *

# collect system arguments
parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="videos", help="directory of the videos")
parser.add_argument("--extension", "-ex",  help="Video file extension")
parser.add_argument("--algorithm", "-fd", help="Feature detection algorthim", choices=['SIFT', 'ORB'])
args = parser.parse_args()
directory = args.directory # "C:/Users/razek/Desktop/file"
fd_algorithm = args.algorithm #"SIFT"
video_extension = args.extension

files_directory, videos_names = find_videos(directory, video_extension)

def save_frames(vid, vid_name, fd_algorithm, final_frames_ids):
    print(f"Saving frames from {vid_name}")
    vidcap = cv2.VideoCapture(vid)
    frame_count = vidcap.get(7)
    count = 0
    while True:
        _, image = vidcap.read()
        if (vidcap.get(1)) in final_frames_ids:
            img_name = f"{vid}_{fd_algorithm}/{str(vidcap.get(1))}{vid_name}_{fd_algorithm}.tif" 
            cv2.imwrite(img_name,image)
         #count frames 
        count += 1

        # check if video is fully read 
        if count >= frame_count:
                break 
    vidcap.release()    

def select_frames (vid, vid_name, fd_algorithm, scalar = 5):
    # basic feedback
    print(f"Parsing {vid_name} using {fd_algorithm}")

    # read the video
    vidcap = cv2.VideoCapture(vid)

    # get the video dimensions 
    if vidcap.isOpened():
        width       = vidcap.get(3)
        height      = vidcap.get(4)
        frame_count = vidcap.get(7)
        print ("intial frame count", frame_count)

    
    # set video dimensions
    width       = int(width/scalar)
    height      = int(height/scalar)
    frame_size  = width * height

    # set the feature exctraction method and the matcher
    if fd_algorithm == "ORB": 
        feature_detector    = cv2.ORB_create()
        FLANN_INDEX_LSH     = 6
        index_params        = dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
        search_params       = dict(checks=100)
        feature_matcher     = cv2.FlannBasedMatcher(index_params,search_params)

    elif fd_algorithm == "SIFT":
        feature_detector    = cv2.xfeatures2d.SIFT_create()
        feature_matcher     = cv2.BFMatcher()

    # initiate timer
    start = time.time()

    # initiate counter
    count = 0

    # initiate accumulators
    x_distance_median_accumulator = 0
    y_distance_median_accumulator = 0

    # initiate frame lists
    selected_video_frames = []
    selected_section_frames = []
    intial_selection_count = 0
    while True:
            # read the video and convert image to gray
            _, image = vidcap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height))

            # initiate the check by reading the first frame
            if count < 1:
                    currentframe = gray.copy()
                    kp2, des2 = feature_detector.detectAndCompute(currentframe,None)        

            # use the next frames based on the condition below
            if count % 1 == 0:

                    # copy the parameters of the previous frame
                    pastframe = currentframe.copy()
                    kp1, des1 = kp2.copy(), des2.copy()
                    currentframe = gray

                    # find the keypoints and descriptors with orb
                    kp2, des2 = feature_detector.detectAndCompute(currentframe,None) 

                    # apply flann knn matcher
                    matches = feature_matcher.knnMatch(des1,des2,k=2)

                    # ratio test as per Lowe's paper                
                    good_matches = []
                    for i, match_pair in enumerate(matches):
                        #check if the matches consists of a pair of matches and not just one
                        if len(match_pair) == 2: 
                            if match_pair[0].distance < 0.7 * match_pair[1].distance:
                                good_matches += [match_pair[0], match_pair[1]]

                    # initialize distances lists
                    x_distance = []
                    y_distance = []


                    # append distances to their lists
                    for mat in good_matches:
                        x_distance.append((kp1[mat.queryIdx].pt)[0] - (kp2[mat.trainIdx].pt)[0])
                        y_distance.append((kp1[mat.queryIdx].pt)[1] - (kp2[mat.trainIdx].pt)[1])

                    # accumulate the distance 
                    x_distance_median_accumulator += np.median(x_distance)
                    y_distance_median_accumulator += np.median(y_distance)

                    # calculate the intersection ratio
                    intersection_pixels = (width - abs(x_distance_median_accumulator)) * (height - abs(y_distance_median_accumulator))
                    intersection_percent = (intersection_pixels / frame_size) * 100 

                    # add frames which has intersection ratios between 80% and 85% to the section 
                    if 90 < intersection_percent < 95:
                            selected_section_frames.append((vidcap.get(1),cv2.Laplacian(gray, cv2.CV_64F).var()))
                            intial_selection_count += 1



                    # if intersection goes below 80% reset the intersection calculation and add selected frames to the main video selected frames 
                    elif intersection_percent < 90:
                            # print ("len of selected_section_frames", len(selected_section_frames))
                            x_distance_median_accumulator = 0
                            y_distance_median_accumulator = 0
                            selected_video_frames.append(selected_section_frames)
                            selected_section_frames = []


                    if count % 100 == 0:
                        print ("current count: ", count)
                    # print ( "len of selected_video_frames", len (selected_video_frames))

            #count frames 
            count += 1

            # check if video is fully read 
            if count >= frame_count:
                break 

    vidcap.release()
    
    # select best quality frames
    final_frames_ids = []
    for frame_list in selected_video_frames:
        final_frames_ids.append(max(frame_list,key=itemgetter(1))[0])
    
    #create folder for the selected frames
    if not os.path.exists(f"{vid}_{fd_algorithm}"):
        os.mkdir(f"{vid}_{fd_algorithm}")
    
    # save images
    save_frames(vid, vid_name, fd_algorithm, final_frames_ids)   
    
    end = time.time() - start
    print ('processing time: ', end)
    return (end, len (final_frames_ids), intial_selection_count )

time_list = []
final_frame_count_list = []
intial_selection_list = []

for count in range(len(files_directory)):
    process_time, final_frame_count, intial_selection = select_frames (files_directory[count], videos_names[count], fd_algorithm)
    time_list.append(process_time)
    final_frame_count_list.append(final_frame_count)
    intial_selection_list.append(intial_selection)

df = pd.DataFrame({
    "video name" : videos_names,
    "processing_time" : time_list,
    "final_frame_count" : final_frame_count_list,
    "intial_selection_count" : intial_selection_list
})

df.to_csv(f"{directory}/{fd_algorithm}.csv")

print ("Done!")