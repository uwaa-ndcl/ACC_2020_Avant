import pkgutil
import subprocess
import numpy as np
import transforms3d as t3d

import net_filter.directories as dirs

class RenderProperties:

    def __init__(self):
        # name of .blend file
        self.model_name = None

        # list of names of images, if none then images will be named
        # 000000.png, 000001.png, ...
        self.image_names = None

        # directory to save output images, etc.
        self.save_dir = None # this will be filled later

        # object
        self.ob = None # this will be set inside of Blender
        self.n_renders = 1
        self.pos = np.array([[0],[0],[0]]) # size (3, n_renders)
        self.quat = np.array([[1],[0],[0],[0]]) # size (4, n_renders)

        # world lighting, size (3, n_renders)
        self.world_RGB = None  

        # lighting energy (sometimes called power in Blender) of all lights
        self.lighting_energy = None

        # transparent background?
        self.alpha = False 

        # camera
        self.cam_ob = None # this will be set inside of Blender
        self.cam_pos = [0, 0, 0]
        self.cam_quat = t3d.euler.euler2quat(np.pi/2, 0, 0, axes='sxyz')
        self.pix_width = 640
        self.pix_height = 480
        self.sensor_fit = 'AUTO'
        self.angle_w = 2*np.arctan(18/50) # Blender default
        self.angle_h = 2*np.arctan(18/50)


def blender_render(render_dir):
    '''
    call a blender command which will generate renders in render_dir
    '''

    # get path to render script
    mod_name = 'net_filter.blender.process_renders'
    pkg = pkgutil.get_loader(mod_name)
    render_script = pkg.get_filename()

    # run blender command
    blender_cmd = 'blender --background --python-use-system-env --python ' \
                  + render_script + ' -- ' + render_dir
    subprocess.run([blender_cmd], shell=True)
