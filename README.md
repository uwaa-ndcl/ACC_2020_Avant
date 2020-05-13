# TITLE!!!

![Image of Chairs](images/fig4.png)

This is the code for the paper

[**Observability Properties of Object Pose Estimation**](https://ieeexplore.ieee.org/document/8814791)

Trevor Avant & Kristi A. Morgansen, *American Control Conference (ACC) 2020*

## before running the code

Before running this code you should install [Blender](https://www.blender.org), add this repository to your `PYTHONPATH` (because the `pose_estimation` directory is structured as a Python package), and change the directories in `pose_estimation/directories.py` to directories on your computer.

software (version used): Python (3.8), Blender (2.82)
python packages (not part of standard library): numpy, scipy, torch, torchvision, cv2, imageio, PIL, pyrr, transforms3d, pyyaml

Download the soup can checkpoint file (`soup_60.pth`) from the authors of Dope from [here](https://drive.google.com/drive/folders/1DfoA3m_Bm0fW8tOWXGVxi4ETlLEAgmcg). Put that file in `/net_filter/dope/`. Note that file is linked from the [Dope Github page](https://github.com/NVlabs/Deep_Object_Pose).

## code

**1-D Motions (Figure 3)**:
Run the following commands:
* `python net_filter/sim/one_dim_gen.py` (generate the images using Blender and evaluate them using Dope, and plot the results)

**Covariances (Figure 4)**:
Run the following commands:
* `python net_filter/sim/cov_gen.py` (generate the images using Blender and evaluate them using Dope)
* `python net_filter/sim/cov_calc.py` (calculate the covariances from the Dope estimates)

**Simulation (Figure 5)**: 
Run the following commands:
* `python net_filter/sim/dynamic_gen.py` (generate the images using Blender)
* `python net_filter/sim/dynamic_eval.py` (evaluate the images using the Dope estimator)
* `python net_filter/sim/dynamic_filter.py` (apply the filter to the Dope estimates)


## Blender models
