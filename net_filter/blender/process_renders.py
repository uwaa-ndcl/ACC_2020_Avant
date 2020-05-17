'''
this script is to be run as a blender command:

blender --background --python script_name.py -- data_dir

data_dir: directory containing to_render.pkl, which should contain a
          RenderProperties object
'''
import bpy
import os.path
import sys
import math
import pickle

import net_filter.directories as dirs
import net_filter.blender.functions as bf

# get arguments, see:
# https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after "--"
data_dir = argv[0]

# load data from pkl file
to_render_pkl = os.path.join(data_dir, 'to_render.pkl')
with open(to_render_pkl, 'rb') as input:
    render_props = pickle.load(input)
render_props.save_dir = data_dir

# model info
blender_models_dir = dirs.blender_models_dir
blend_file = os.path.join(blender_models_dir, render_props.model_name+'.blend')
bpy.ops.wm.open_mainfile(filepath=blend_file)
ob_name = 'all_parts'
ob = bpy.data.objects[ob_name]
render_props.ob = ob

# camera
camera_name = 'cam0'
cam = bpy.data.cameras.new(camera_name)  # create a new camera
cam_ob = bpy.data.objects.new(camera_name, cam)  # create a new camera object
bpy.context.scene.camera = cam_ob  # set the active camera
bpy.data.objects[camera_name].location = render_props.cam_pos
cam_ob.rotation_mode = 'QUATERNION'
bpy.data.objects[camera_name].rotation_quaternion = render_props.cam_quat
render_props.cam_ob = cam_ob

# lens and sensor
f = 50 # focal length, Blender default
bpy.data.cameras[camera_name].lens = f
# set the sensor size to produce the correct angle
# (Blender doesn't work too well when you try to set the angle itself)
bpy.data.cameras[camera_name].sensor_fit = render_props.sensor_fit
bpy.data.cameras[camera_name].sensor_width = 2*f*math.tan(render_props.angle_w/2)
bpy.data.cameras[camera_name].sensor_height = 2*f*math.tan(render_props.angle_h/2)

# scene properties
bpy.data.scenes['Scene'].render.resolution_x = render_props.pix_width
bpy.data.scenes['Scene'].render.resolution_y = render_props.pix_height
bpy.data.scenes['Scene'].render.resolution_percentage = 100
bpy.data.scenes['Scene'].render.image_settings.file_format = 'PNG'
bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
bpy.data.scenes['Scene'].cycles.film_transparent = True
bpy.data.scenes['Scene'].render.engine = 'CYCLES'
#bpy.data.scenes['Scene'].render.engine = 'BLENDER_EEVEE'
bpy.data.scenes['Scene'].cycles.device = 'GPU'

# render pose
bf.render_pose(render_props)
