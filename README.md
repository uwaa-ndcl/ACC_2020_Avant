# Rigid Body Dynamics Estimation by Unscented Filtering Pose Estimation Neural Networks

![Image of Chairs](images/fig4.png)

This is the code for the paper

[**Rigid Body Dynamics Estimation by Unscented Filtering Pose Estimation Neural Networks**]()

Trevor Avant & Kristi A. Morgansen, *American Control Conference (ACC) 2020*

## before running the code

Before running this code you should install [Blender](https://www.blender.org), add this repository to your `PYTHONPATH` (because the `pose_estimation` directory is structured as a Python package), and change the directories in `pose_estimation/directories.py` to directories on your computer.

software (version used): Python (3.8), Blender (2.82)
python packages (not part of standard library): numpy, scipy, torch, torchvision, cv2, imageio, PIL, pyrr, transforms3d, pyyaml

Download the soup can checkpoint file (`soup_60.pth`) from the authors of Dope from [here](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg). Put that file in `/net_filter/dope/`. Note that file is linked from the [Dope Github page](https://github.com/NVlabs/Deep_Object_Pose).

## code

**1-d motions (Figure 3)**:
Run the following commands:
* `python net_filter/sim/one_dim_gen.py` (generate the images using Blender and evaluate them using Dope, and plot the results)

**covariances (Figure 4)**:
Run the following sequence of commands twice: once with `mode = dim` and once with `mode = bright` uncommented at the top of each (each sequence generates 1,000 images, and takes about 30 min on my computer):
* `python net_filter/sim/cov_gen.py` (generate the images using Blender and evaluate them using Dope)
* `python net_filter/sim/cov_calc.py` (calculate the covariances from the Dope estimates)

**simulation (Figure 5 \& Figure 6)**:
Run the following command:
* `python net_filter/sim/dynamic_run_once.py`

**randomized simulations (Table I)**: 
Run the following commands:
* `python net_filter/sim/dynamic_run_n_times.py`
* `python net_filter/sim/dynamic_run_n_times_results.py`

## Blender model
In this paper, we considered our object of interest to be the soup can from the YCB dataset. The blender model we used is located in the directory `blender_models/soup_can.blend`. This model was created by downloading a laser scan model from [this page](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/), opening it in Blender, and converting it to Blender's `.blend` format.


## extra

**make the GIF**
Run the following commands (requires Imagemagick to be installed):
* `python net_filter/sim/box_video_comparison.py`
