import argparse
import cv2 
from operator import itemgetter
import pandas as pd
import os
import time

from utilities import *


# collect system arguments
parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", default="videos", help="directory of the videos")
parser.add_argument("--extension", "-ex",  help="Video file extension")
parser.add_argument("--nth", help="Interval between frames", type=int, default=25)
args = parser.parse_args()
directory = args.directory 
video_extension = args.extension
nth_interval = int(args.nth)

files_directory, videos_names = find_videos(directory, video_extension)

def select_frames (vid, vid_name, nth_interval):
	print(f"Saving frames from {vid_name}")
	start = time.time()
    
	# read the video
	vidcap = cv2.VideoCapture(vid)
    
    #create folder for the selected frames
	if not os.path.exists(f"{vid}_bl{nth_interval}"):
		os.mkdir(f"{vid}_bl{nth_interval}")

    # get the video dimensions 
	if vidcap.isOpened():
		frame_count = vidcap.get(7)
		print (f"intial frame count: {int(frame_count)}")


	# initiate counter
	count = 0
	selected_frames_count = 0
    
    # parse the video
	while True:
            # read the video
		success, image = vidcap.read()
            
		if vidcap.get(1) == 0:
			cv2.imwrite(f"{vid}_bl{nth_interval}/{str(vidcap.get(1))}{vid_name}_bl{nth_interval}.tif",image)

            # use the next frames based on the condition below
		if vidcap.get(1) % nth_interval == 0:
			cv2.imwrite(f"{vid}_bl{nth_interval}/{str(vidcap.get(1))}{vid_name}_bl{nth_interval}.tif",image)
			selected_frames_count += 1

            # check if video is fully read 
		if vidcap.get(1) >= frame_count:
			break 

	vidcap.release()
     
    
	end = time.time() - start
	print ('Processing time: ', end)
	print ('Final frame count: ', selected_frames_count)
	return (end, selected_frames_count)

time_list = []
final_frame_count_list = []

for count in range(len(files_directory)):
	process_time, frame_count = select_frames (files_directory[count], videos_names[count], nth_interval)
	time_list.append(process_time)
	final_frame_count_list.append(frame_count)

df = pd.DataFrame({
	"video name" : videos_names,
	"processing_time" : time_list,
	"final_frame_count" : final_frame_count_list, 
})

df.to_csv(f"{directory}/baseline_{nth_interval}.csv")

print ("Done!")