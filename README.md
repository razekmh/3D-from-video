<a href="https://github.com/razekmh/3D-from-video"><img src="https://github.com/razekmh/3D-from-video/blob/master/media/3d_from_video.png" title="3D from video" alt="3D from video"></a>
# 3D-from-video
> Optimum frame selection for 3D reconstruction from video


![APM](https://img.shields.io/apm/l/vim-mode?style=flat-square)
![PowerShell Gallery](https://img.shields.io/powershellgallery/p/DNS.1.1.1.1?style=flat-square)
![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/metabolize/rq-dashboard-on-heroku?style=flat-square)


<!-- > photogrammetry, video, visual odometry, keypoint detection -->


## Table of Contents
- [Prerequisites](#Prerequisites)
- [Installation](#installation)
- [Features](#features)
- [Usage](#Usage)
- [Theory](#Theory)
- [Contributing](#contributing)
- [Team](#team)
<!-- - [FAQ](#faq)
- [Support](#support)
- [License](#license) -->

## Prerequisites

- python 3.* 
- opencv 3.4.2.16
- opencv-contrib 3.4.2.16
- pandas > 0.25.*

You can install the libraries individually or using the requiremnts.txt included in the repository. 


## Installation

- first clone the repository
```shell
git clone https://github.com/razekmh/3D-from-video.git
```

- then install the requirements if not already installed  
```shell
pip install -r requirement.txt
```

## Features

- Produces a folder with extracted frames for each video
- Parses a folder of videos on one go 
- Supports three different methodologies and four varieties of algorithms

## Usage

To use the modules, you will need at least one video file. Start by placing the video in a folder and keep track of the directory as you will need it in using each of the modules. The default directory for the videos is a folder tilted ```videos``` in the same directory as the modules is saved. If you choose to save your videos in such folders, then you don't need to pass the directory argument to the module while running it. In all the modules the directory argument can be passed using ```-d``` as in example below

```shell
python vo_main.py -d directory
```

The modules also assume that the video extension is ```MOV``` unless you specify the extension in the arguments. In all the modules the extension argument can be passed using ```-ex``` as in example below

```shell
python vo_main.py -ex extension
```

### The algorthim provides four main options
- [Baseline](#Baseline)
- [Feature Displacement (SIFT detector and BruteForce matcher)](#SIFT)
- [Feature Displacement (ORB detector and FLANN matcher)](#ORB)
- [Visual Odometry](#Odometry)



### Baseline
This option allows you to extract frames from video on regular intervales. It is included in the module ```nth_main.py```. The modules also assume that the intervales are ```25``` frames  unless you specify it in the arguments. The interval argument can be passed using ```-nth``` as in example below

```shell
python nth_main.py -nth interval
``` 

### SIFT
This option allows you to extract frames based on feature displacement algorithm while using SIFT detector and BruteForce matcher. It is included within the module ```fd_main.py```. To use SIFT you need to specify the ```-fd``` argument by typing ```SIFT``` as in the example below

```shell
python fd_main.py -fd SIFT
``` 

### ORB
This option allows you to extract frames based on feature displacement algorithm while using ORB detector and FLANN matcher. It is included within the module ```fd_main.py```. To use SIFT you need to specify the ```-fd``` argument by typing ```ORB``` as in the example below

```shell
python fd_main.py -fd ORB
```

### Odometry
This option allows you to extract frames based on visual odometry algorithm. It requires more parameters to operate properly. The additional parameters should be provided in the for of a ```xml``` file titled after the video with ```_camera_calibration.xml``` trailing after, as in the following example

```
video_name_camera_calibration.xml
```
The structure and information included in the video calibration are included in an example file <a href="https://github.com/razekmh/3D-from-video/blob/master/media/video_name_camera_calibration.xml">here</a>


### Output
All modules will export the selected frames in a folder named after the video with a suffix pointing to the module/algorithm used. Additionally basic statistics will be exported in simple text file. 


## Theory
<img src="https://github.com/razekmh/3D-from-video/blob/master/media/outline.png" title="Study outline" alt="Study outline"></a>



## Contributing

> To get started...
### Step 1
- Show ❤️! ⭐️ the repository
    
### Step 2

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine using `https://github.com/razekmh/3D-from-video.git`

### Step 3

- **HACK AWAY!** 🔨🔨🔨

### Step 4

- 🔃 Create a new pull request using <a href="https://github.com/razekmh/3D-from-video/compare/" target="_blank">`https://github.com/razekmh/3D-from-video/compare/`</a>.

---