'''
these are functions to be called from within Blender using Blender's python
interface
'''
import os
import bpy
import numpy as np

import net_filter.tools.image as ti


def render_image(cam_ob, cam_pos, cam_quat, ob, ob_pos, ob_quat, image_file,
                 alpha=True, world_RGB=None):
    '''
    set the camera and object to a position and orientation, take, and save an
    image
    '''

    # camera properties
    cam_ob.location = cam_pos
    cam_ob.rotation_mode = 'QUATERNION'
    cam_ob.rotation_quaternion = cam_quat

    # object properties
    ob.location = ob_pos
    ob.rotation_mode = 'QUATERNION'
    ob.rotation_quaternion = ob_quat

    # save file
    bpy.data.scenes['Scene'].render.filepath = image_file

    # color of the "world"
    if world_RGB is not None:
        A = np.array([1.0]) # alpha for world RGBA lighting
        RGBA = np.concatenate((world_RGB, A))
        bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value = RGBA

    # transparent background?
    if not alpha:
        bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGB'
        bpy.data.scenes['Scene'].render.film_transparent = False
    else:
        bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
        bpy.data.scenes['Scene'].render.film_transparent = True

    bpy.ops.render.render(write_still=True)


def render_pose(render_props):
    '''
    render the object at different x, y, and z locations and orientations (as
    quaternions)
    '''

    # preliminary things
    if render_props.image_names is None:
        image_numerical_name = '%06d.png' # generic name for each image

    if render_props.lighting_energy is not None:
        for light in bpy.data.lights:
            light.energy = render_props.lighting_energy

    # loop through poses to generate images
    for i in range(render_props.n_renders):

        # different world color?
        if render_props.world_RGB is not None:
            world_RGB_i = render_props.world_RGB[:, i]
        else:
            world_RGB_i = None
       
        # give the image a name
        if render_props.image_names is None:
            image_file_name_i = image_numerical_name % i
        else:
            image_file_name_i = render_props.image_names[i]
        image_file_i = os.path.join(render_props.save_dir, image_file_name_i)
        
        # render image i
        render_image(
            cam_ob=render_props.cam_ob,
            cam_pos=render_props.cam_pos,
            cam_quat=render_props.cam_quat,
            ob=render_props.ob,
            ob_pos=render_props.pos[:,i],
            ob_quat=render_props.quat[:,i],
            image_file=image_file_i,
            alpha=render_props.alpha,
            world_RGB=world_RGB_i)
