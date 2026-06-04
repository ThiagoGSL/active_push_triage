import os
import numpy as np
from gymnasium import envs
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--numImages", type=int, default=24000, help="num_images are generated (num_images = num_train_images + num_test_images)")
parser.add_argument("--fixedObjectHeight", type=lambda x: None if x == 'None' else float(x), default=None, help="object height")
parser.add_argument("--useSimConfig", type=int, choices=[0, 1], default=1, help="use sim or real camera config?")
config = parser.parse_args()

# params
data_path = os.getenv("UR3_PUSH_DATAPATH")
seed = 100 # seed used to generate images
image_size = 64 # height = width

# set desired object/target params 
object_reset_options = { 
                        "obj_type": None, 
                        "obj_mass": -1, 
                        "obj_sliding_friction": -1,
                        "obj_torsinal_friction": -1,
                        "obj_size_0": None,
                        "obj_size_1": None,
                        "obj_size_2": None,
                        "obj_xy_pos": None,      
                        "obj_quat": None,
                        "target_xy_pos": None,
                        "target_quat": None,   
                        }

try:
    # gym environment
    env_kwargs = {
            "render_mode": "rgb_array",
            "object_reset_options": object_reset_options,
            "object_params": {"range_x_pos":np.array([-0.25,0.3]), "range_y_pos":np.array([-0.3,0.3])} if config.useSimConfig else {"range_x_pos":np.array([-0.23,0.23]), "range_y_pos":np.array([-0.15,0.27])},
            "fixed_object_height": config.fixedObjectHeight,
            "latent_dim": None,
            "camera_options": {"camera_name": "rgb_cam", "width": image_size, "height": image_size},
            "use_sim_config": config.useSimConfig
        } 
    env_str = 'MujocoUR3PushEnv'

    env = envs.make(env_str, **env_kwargs)
    env.unwrapped.threshold_obj_start_pose = 1e-7
    observation,_ = env.reset(seed=seed)

    # generate images
    img_height = env.unwrapped.image_height
    img_width = env.unwrapped.image_width
    binary_images = np.zeros((config.numImages, img_height, img_width), dtype=np.uint8)

    # fig, axs = plt.subplots(1,1)
    for i in range(0, config.numImages):
        if i%500 == 0 and i>0:
            print(f"number of generated images: {i}/{config.numImages}")

        # sample new object and target params
        env.reset(options=object_reset_options)
        # use target image as it is already generated during reset
        binary_images[i,:,:] = env.unwrapped.target_binary_image
        
        # axs.imshow(binary_images[i,:,:], cmap="gray")
        # plt.draw()
        # plt.pause(0.0001)
        # stop = 0

    env.close()
    
except KeyboardInterrupt:
    pass
else:
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    if not os.path.isdir(os.path.join(data_path, "data")):
        os.mkdir(os.path.join(data_path, "data"))
    if not os.path.isdir(os.path.join(data_path, "plots")):
        os.mkdir(os.path.join(data_path, "plots"))

    # save image data
    image_file = os.path.join(data_path, "data", f"images_imgHeight_{img_height}_imgWidth_{img_width}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{config.useSimConfig}.npy")
    with open(image_file, "wb") as f:
        np.save(f, binary_images)

    # plot some images
    fig, axs = plt.subplots(3,3, figsize=(10,10))
    fig.suptitle("Example Binary Images", fontsize=18)
    for i in range(0,axs.shape[0]):
        for j in range(0,axs.shape[1]):
            axs[i,j].imshow(binary_images[i*axs.shape[1]+j,:,:], cmap="gray")
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    plt.savefig(os.path.join(data_path, "plots", f"example_images_imgHeight_{img_height}_imgWidth_{img_width}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{config.useSimConfig}.png"))