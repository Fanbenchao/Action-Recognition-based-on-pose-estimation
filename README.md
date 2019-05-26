<!DOCTYPE html>
<html>
<body>
<h1>Action Recognition based on pose estimation</h1>
  <p>We repeat the results of the follwing CVPR 2018 paper:
    <b><br>2D/3D Pose Estimation and Action Recognition using Multitask Deep leaning</br></b>
    <br>if you'd like to refer to the original codes, <a href="https://github.com/dluvizon/deephar">Please tap this link </a> </br>
    <br>our contribution is to supplement the original codes, 
    such as the training process of the 3D pose estimation and action recognition etc.</br>
  </p>
<h2>Details</h2>
  <p><b>language: </b>python3.6
    <br><b>frame: </b>tensorflow 1.10+/keras 2.1.4</br>
    <br><b>GPU: </b>single GPU</br>
  </p>
<h2>Datasets</h2>
  <p><a href = "http://human-pose.mpi-inf.mpg.de/"><b>MPII dataset: </b></a>
    We use this dataset train 2D pose estimation like the paper.
    <br><a href = "http://rose1.ntu.edu.sg/datasets/actionrecognition.asp"><b>NTU RGB-D dataset: </b></a>
      We use this dataset train 3D pose estimation and action recognition, due to cannot download Penn action and Human3.6 dataset.
    </br>
   </p>
<h2>Data Process and Visulization</h2>
  <p><b>data_generator/annotation_process.py: </b> the process of mpii dataset.
    <br><b>data_generator/video_clip.ipynb: </b> transform the videos in ntu dataset to rgb images.</br>
    <br><b>data_generator/image_show.ipynb: </b> draw 2D skeletons in images</br>
    <br><b>data_generator/3D_pose_imgShow.ipynb: </b> draw 2D skeletons in images and 3D skeletons spatial map.</br>
  </p>
<h2>Training Process</h2>
  <p><b>2D pose estimation: </b>
      Just run pose estimation.ipynb.
    <br><b>3D pose estimation: </b> Just run 3d_pose.ipynb.</br>
    <br><b>Action reconition: </b>Just run 3d_pose.ipynb.</br>
  </p>
</body>
</html>
