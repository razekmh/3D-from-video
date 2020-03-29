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

args = parser.parse_args()
directory = args.directory 
video_extension = args.extension

files_directory, videos_names = find_videos(directory, video_extension)

# 
STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1000

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict(winSize  = (21, 21), 
                maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def featureTracking(image_ref, image_cur, kp_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, kp_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
    st = st.reshape(st.shape[0])
    kp1 = kp_ref[st == 1]
    kp2 = kp2[st == 1]
    
    return kp1, kp2


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.kp_ref = None
        self.kp_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.detector = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=False)

    def processFirstFrame(self):
        self.kp_ref = self.detector.detect(self.new_frame)
        self.kp_ref = np.array([x.pt for x in self.kp_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME
        
    def processSecondFrame(self):
        self.kp_ref, self.kp_cur = featureTracking(self.last_frame, self.new_frame, self.kp_ref)
        E, mask = cv2.findEssentialMat(self.kp_cur, self.kp_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=3.0)
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.kp_cur, self.kp_ref, focal=self.focal, pp = self.pp)
        self.frame_stage = STAGE_DEFAULT_FRAME 
        self.kp_ref = self.kp_cur

    def processFrame(self, frame_id):
        self.kp_ref, self.kp_cur = featureTracking(self.last_frame, self.new_frame, self.kp_ref)
        E, mask = cv2.findEssentialMat(self.kp_cur, self.kp_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, self.kp_cur, self.kp_ref, focal=self.focal, pp = self.pp)
        self.cur_t = self.cur_t + self.cur_R.dot(t)
        self.cur_R = self.cur_R.dot(R)
        
        
        if(self.kp_ref.shape[0] < kMinNumFeature):
            self.kp_cur = self.detector.detect(self.new_frame)
            self.kp_cur = np.array([x.pt for x in self.kp_cur], dtype=np.float32)
        
        self.kp_ref = self.kp_cur
        
    def update(self, img, frame_id):
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        
        if(self.frame_stage == STAGE_DEFAULT_FRAME):
            self.processFrame(frame_id)
        elif(self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame()
        elif(self.frame_stage == STAGE_FIRST_FRAME):
            self.processFirstFrame()
        self.last_frame = self.new_frame

def select_frames (vid, vid_name):
    # basic feedback
    print(f"Parsing {vid_name}")

    # start timer
    start = time.time()

    # create video object
    vidcap = cv2.VideoCapture(vid)

    # get basic measures 
    width = vidcap.get(3)
    height = vidcap.get(4)
    org_frame_count = int(vidcap.get(7))

    print ('Initial frame_count is ', org_frame_count)
    
    resize_factor = 4
    width = int(width/resize_factor)
    height = int(height/resize_factor)
    frame_size = width * height

    # selection grid_size 
    grid_size = 5

    # get calibration parameters
    distCoeff = np.zeros((4,1),np.float64)
    parameters_dict = get_parameters(vid)
    k1 = parameters_dict['k1']  #distortion_coefficient; 
    k2 = parameters_dict['k2'] 
    p1 = parameters_dict['p1'] 
    p2 = parameters_dict['p2'] 

    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;

    # if focal length is not provided it is guessed as average of width and height
    if parameters_dict['f'] == 0:
        focal_length = (vidcap.get(3)+vidcap.get(4))/2
    else: 
        focal_length = parameters_dict['f']

    # assume unit matrix for camera
    cam_mat = np.eye(3,dtype=np.float32)

    cam_mat[0,2] = width/2.0  # define center x
    cam_mat[1,2] = height/2.0 # define center y
    cam_mat[0,0] = focal_length        # define focal length x
    cam_mat[1,1] = focal_length        # define focal length y

    # set graphics parameters
    traj_amplification = 5
    traj_width = 1200
    traj_height = 1200
    
    # list coordinations
    traj_z = np.full((traj_width,traj_height,3),255, dtype=np.uint8)
    traj_y = np.full((traj_width,traj_height,3),255, dtype=np.uint8)
    
    # intialize counter and collection lists
    x_list, y_list, z_list , sharp_list, frame_list  = [], [], [], [], [] 
    count = -1

    #create folder for the selected frames
    create_folder_for_video(vid, '_vo')
    
    # create camera object
    cam = PinholeCamera(float(width), float(height), focal_length, focal_length, float(width)/2, float(height)/2)
    vo = VisualOdometry(cam)

    while True:
            # parse the video
            success, image = vidcap.read()
            
            count += 1
            if count >= org_frame_count:
                break
            if count % 1 != 0: 
                continue
            
            # read the video and convert image to gray
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height))

            # clacluate sharpness and undistort
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            dst = cv2.undistort(gray,cam_mat,distCoeff)
            
            vo.update(dst, count)
            cur_t = vo.cur_t
            if(count > 2):
                x, y, z = cur_t[0], cur_t[1], cur_t[2]
            else:
                x, y, z = 0., 0., 0.

            try: 
                x_list.append(int(grid_size * round(x[0]/grid_size)))
                y_list.append(int(grid_size * round(y[0]/grid_size)))
                z_list.append(int(grid_size * round(z[0]/grid_size)))
                sharp_list.append(sharp)
                frame_list.append(vidcap.get(1))
            except:
                pass
            
            # draw movement on the trajectory window of (x,z) and update output graphics
            draw_x, draw_y = int((x*traj_amplification)+(traj_width/2)), int((z*traj_amplification)+(traj_height/2))
            cv2.circle(traj_z, (draw_x,draw_y), 1, (255-((vidcap.get(1)/org_frame_count)*255),0,((vidcap.get(1)/org_frame_count)*255)), 1)
            cv2.rectangle(traj_z, (10, 20), (600, 60), (255,255,255), -1)
            text = "Coordinates: x=%2fpx y=%2fpx z=%2fpx"%(x,y,z)
            cv2.putText(traj_z, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, 8)
            # 
            cv2.imwrite(vid+'_vo/'+vid_name+"traj_z.png",traj_z)


            # draw movement on the trajectory window of (x,y) and update output graphics
            draw_x, draw_y = int((x*traj_amplification)+(traj_width/2)), int((y*traj_amplification)+(traj_height/2))
            cv2.circle(traj_y, (draw_x,draw_y), 1, (0,255-((vidcap.get(1)/org_frame_count)*255),((vidcap.get(1)/org_frame_count)*255)), 1)
            cv2.rectangle(traj_y, (10, 20), (600, 60), (255,255,255), -1)
            text = "Coordinates: x=%2fpx y=%2fpx z=%2fpx"%(x,y,z)
            cv2.putText(traj_y, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, 8)
            cv2.imwrite(vid+'_vo/'+vid_name+"traj_y.png",traj_y)
             
            cv2.waitKey(1)

            # display the results while parsing 
            cv2.imshow('Trajectory_Z', traj_z)
            cv2.imshow('Trajectory_Y', traj_y)
            cv2.imshow('Drone Camera', gray)

            # check if video is fully read
            if count >= org_frame_count:
                break

    vidcap.release()
    cv2.destroyAllWindows()
    
    #create the grid
    df = pd.DataFrame({
    "ID" : frame_list,
    "x" : x_list,
    "y" : y_list,
    "z" : z_list, 
    "sharp" : sharp_list
    })
    
    df_group = df.groupby(['x', 'z'])

    final_list = []
    for i, j in df_group:
        final_list.append(j.ID[j.idxmax(axis= 0)['sharp']])
        
    
    
    # save images
    vidcap = cv2.VideoCapture(vid)
    count = 0
    while True:
        success, image = vidcap.read()
        if (vidcap.get(1)) in final_list:
            cv2.imwrite(vid+'_vo/'+str(vidcap.get(1))+vid_name+"_vo.tif",image)
         #count frames 
        count += 1

        # check if video is fully read 
        if count >= org_frame_count:
                break 
    vidcap.release()    
    
    end = time.time() - start
    print ('time: ', end)
    return (end, len (final_list), org_frame_count)


# write output stats
def wrtie_header(path):
    str_r = 'video name;time;frame_count;org_frame_count'
    f = open(os.path.join(path,"VO_fast_selection.txt"), "w")
    f.write(str_r)
    f.close()

def write_output(path, video_name, time, frame_count, org_frame_count):
    str_r = '\n' + str(video_name) + ';' +  str(time) + ';' + str(frame_count) + ';' + str (org_frame_count)
    f = open(os.path.join(path,"VO_fast_selection.txt"), "a")
    f.write(str_r)
    f.close()

wrtie_header(directory)
for count in range(len(files_directory)):
    process_time, frame_count, org_frame_count = select_frames (files_directory[count], videos_names[count])
    write_output(directory, videos_names[count], process_time, frame_count, org_frame_count)

print ("Done!")

