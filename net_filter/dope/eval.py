import yaml
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as pp

import net_filter.directories as dirs
from net_filter.dope.cuboid import *
from net_filter.dope.detector import *
import net_filter.dope.draw as dd


def load(yaml_file, ckpt_file):
    '''
    load the yaml file and model
    '''

    with open(yaml_file, 'r') as stream:
        try:
            print("Loading DOPE parameters from '{}'...".format(yaml_file))
            params = yaml.load(stream, Loader=yaml.FullLoader)
            print('    Parameters loaded.')
        except yaml.YAMLError as exc:
            print(exc)

        models = {}
        pnp_solvers = {}
        draw_colors = {}

        # Initialize parameters
        matrix_camera = np.zeros((3,3))
        matrix_camera[0,0] = params["camera_settings"]['fx']
        matrix_camera[1,1] = params["camera_settings"]['fy']
        matrix_camera[0,2] = params["camera_settings"]['cx']
        matrix_camera[1,2] = params["camera_settings"]['cy']
        matrix_camera[2,2] = 1
        dist_coeffs = np.zeros((4,1))

        if "dist_coeffs" in params["camera_settings"]:
            dist_coeffs = np.array(params["camera_settings"]['dist_coeffs'])
        config_detect = lambda: None
        config_detect.mask_edges = 1
        config_detect.mask_faces = 1
        config_detect.vertex = 1
        config_detect.threshold = 0.5
        config_detect.softmax = 1000
        config_detect.thresh_angle = params['thresh_angle']
        config_detect.thresh_map = params['thresh_map']
        config_detect.sigma = params['sigma']
        config_detect.thresh_points = params["thresh_points"]

        # for each object to detect, load network model, create PNP solver
        for model in params['weights']:
            models[model] = ModelData(model, ckpt_file)
            models[model].load_net_model()

            draw_colors[model] = tuple(params["draw_colors"][model])

            pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    matrix_camera,
                    Cuboid3d(params['dimensions'][model]),
                    dist_coeffs=dist_coeffs
                )

    return models, pnp_solvers, config_detect, draw_colors


def eval(img_files, yaml_file, ckpt_file, draw=True, save_boxed_image=False):
    '''
    evaluate img_files using the Dope network 
    '''

    n_ims = len(img_files)
    exposure_val = 166

    models, pnp_solvers, config_detect, draw_colors = load(yaml_file, ckpt_file)

    xyz_quat = np.full((7,n_ims), np.nan) # xyz and quaternion for all images
    for i in range(n_ims):
        #print('i',i)

        # read image
        img_file_i = img_files[i]
        img = cv2.imread(img_file_i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw_ob = ImageDraw.Draw(im)

        for m in models:
            # detect object
            #detected_objects = ObjectDetector.detect_object_in_image(
            detected_objects, vertex2_np, aff_np, maps = ObjectDetector.detect_object_in_image(
                models[m].net,
                pnp_solvers[m],
                img,
                config_detect
            )
            #print(detected_objects)

            # overlay cube on image
            for i_r, detected_object in enumerate(detected_objects):
                if detected_object["location"] is None:
                    continue
                loc = detected_object["location"]
                ori = detected_object["quaternion"]

                # save
                loc = np.asarray(loc)
                ori = np.asarray(ori)
                xyz_quat[:3,i] = loc 
                xyz_quat[3:,i] = ori 

                # draw the cube
                if None not in detected_object['projected_points']:
                    points2d = []
                    for pair in detected_object['projected_points']:
                        points2d.append(tuple(pair))
                    dd.draw_cube(draw_ob, points2d, color=draw_colors[m])

        open_cv_image = np.array(im)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        if save_boxed_image:
            img_dir_i, tail_i = os.path.split(img_file_i)
            img_boxed_dir_i = os.path.join(img_dir_i, 'boxed')
            img_boxed_i = os.path.join(img_boxed_dir_i, tail_i)
            if not os.path.exists(img_boxed_dir_i):
                os.makedirs(img_boxed_dir_i)
            cv2.imwrite(img_boxed_i, open_cv_image)

        if draw:
            cv2.imshow('Open_cv_image', open_cv_image)
            cv2.waitKey(1)

    return xyz_quat
