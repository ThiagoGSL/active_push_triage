import os
import numpy as np
from copy import deepcopy
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from franka_msgs.msg import FrankaState
import tf.transformations
from ur3_push_rl_sb3.utils import parse_args, get_run_name, get_log_paths
from stable_baselines3.sac.policies import MultiInputPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ur3_push_mujoco.utils import ros_utils, image_utils
from ur3_push_mujoco.autoencoder.VAE import VAE
import cv2 as cv
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from gymnasium import spaces

os.environ["ROS_MASTER_URI"] = "http://orpheus:11311"

# messages
link_name = "panda_link0"
current_msg = {"franka_state": None, "camera": None}
last_timestamps = {"franka_state": 0, "camera": 0}

def new_msg_callback(msg):
    topic = msg._connection_header["topic"]

    key = [k for k in current_msg.keys() if k in topic]
    if not key:
        rospy.logerr(f"No key found for current topic {topic} - current message is ignored")
    elif len(key) > 1:
        rospy.logerr(f"More than one key found for current topic {topic} - current message is ignored")
    else:
        if current_msg[key[0]] is not None:
            if msg.header.stamp < current_msg[key[0]].header.stamp:
                rospy.logerr(f"Detected jump back in time - current message is ignored (topic: {topic})")
                return
            last_timestamps[key[0]] = current_msg[key[0]].header.stamp # remember timestamp of old message
        current_msg[key[0]] = msg

def cameraImage_to_binaryImage():
    rgb_image = bridge.imgmsg_to_cv2(img_msg=current_msg["camera"]) # convert to OpenCV image (rgb8)
    rgb_image = cv.flip(rgb_image, 1)
    # crop
    rgb_image = rgb_image[130:,210:-80]
    binary_image = image_utils.rgbImg_to_binaryImg(rgb_image, min_max_hue_obj, min_max_saturation_obj, min_max_val_obj).astype(np.float32) # to binary image
    # resize
    rgb_image = cv.resize(rgb_image, dsize=[img_wh,img_wh])
    binary_image = cv.resize(binary_image, dsize=[img_wh,img_wh])

    binary_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN,kernel=np.ones((3,3), dtype=np.uint8))
    binary_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE,kernel=np.ones((3,3), dtype=np.uint8))

    binary_image[np.bitwise_and(binary_image!=0, binary_image!=1)] = 0
    return binary_image, rgb_image

def get_new_pose(action, initial_ee_zpos, min_ee_z_pos):
    last_ee_pos = currentFrankaMsg_to_lastEEPos()

    # publish new pose
    new_pose = PoseStamped()
    new_pose.pose.position.x = last_ee_pos[0] + action[0]
    new_pose.pose.position.y = last_ee_pos[1] + action[1]
    new_pose.pose.position.z = max(initial_ee_zpos, min_ee_z_pos) 

    new_quat = tf.transformations.quaternion_about_axis(angle=np.pi/2, axis=(0,0,1)) 
    new_pose.pose.orientation = Quaternion(*new_quat)

    return new_pose

def publish_new_pos(new_pose, panda_pose_pub):
    
    new_pose.header.stamp = rospy.Time.now()
    panda_pose_pub.publish(new_pose)

    return np.array([new_pose.pose.position.x, new_pose.pose.position.y, new_pose.pose.position.z])

def pusblish_start_pos(start_pos, panda_pose_pub, min_ee_z_pos):
    new_pose = PoseStamped()
    new_pose.pose.position.x = start_pos[0]
    new_pose.pose.position.y = start_pos[1]
    new_pose.pose.position.z = max(start_pos[2], min_ee_z_pos) 

    new_quat = tf.transformations.quaternion_about_axis(angle=np.pi/2, axis=(0,0,1)) 
    new_pose.pose.orientation = Quaternion(*new_quat)
    new_pose.header.stamp = rospy.Time.now()

    panda_pose_pub.publish(new_pose)

def currentFrankaMsg_to_lastEEPos():
    last_ee_pose_stamped = ros_utils.FrankaState_to_measured_ee_pose(current_msg["franka_state"])
    last_ee_pos = ros_utils.point_to_nparray(last_ee_pose_stamped.pose.position)
    return last_ee_pos

def wait_for_initial_pose():
    msg = rospy.wait_for_message("franka_state_controller/franka_states", FrankaState)

    ee_pose_stamped_m = ros_utils.FrankaState_to_measured_ee_pose(msg)
    initial_ee_pos = ros_utils.point_to_nparray(ee_pose_stamped_m.pose.position)

    return initial_ee_pos

def plot_start_goal(goal_binary_image):
    _, axs = plt.subplots(1,3, figsize=(20,10))
    
    while not rospy.is_shutdown():
        print("plotting...")
        object_binary_image, _ = cameraImage_to_binaryImage()
        
        assert np.bitwise_or(object_binary_image == 1, object_binary_image == 0).all()
        assert np.bitwise_or(goal_binary_image == 1, goal_binary_image == 0).all()

        mask_obj = object_binary_image == 1
        mask_goal = goal_binary_image == 1

        img_obj_hsv = cv.cvtColor(cv.cvtColor(object_binary_image, cv.COLOR_GRAY2RGB), cv.COLOR_RGB2HSV)
        img_goal_hsv = cv.cvtColor(cv.cvtColor(goal_binary_image, cv.COLOR_GRAY2RGB), cv.COLOR_RGB2HSV)

        img_obj_hsv[mask_obj, 0] = 180
        img_obj_hsv[mask_obj, 1] = 255
        img_obj_hsv[mask_obj, 2] = 255

        img_goal_hsv[mask_goal, 0] = 150
        img_goal_hsv[mask_goal, 1] = 255
        img_goal_hsv[mask_goal, 2] = 255

        img_obj_rgb = cv.cvtColor(img_obj_hsv, cv.COLOR_HSV2RGB)
        img_goal_rgb = cv.cvtColor(img_goal_hsv, cv.COLOR_HSV2RGB)

        img_rgb = img_goal_rgb.copy()
        img_rgb[mask_obj, :] = img_obj_rgb[mask_obj, :].copy()

        axs[0].cla()
        axs[0].imshow(img_obj_rgb)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_title("object")

        axs[1].cla()
        axs[1].imshow(img_goal_rgb)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_title("goal")

        axs[2].cla()
        axs[2].imshow(img_rgb)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_title("object and goal")

        plt.draw()
        plt.pause(0.0001)
        plt.savefig('test' + ".png", bbox_inches="tight")

if __name__ == "__main__":
    # input args that determine dir name of best model
    config, cmd_args, config_parser = parse_args()
    run_name = get_run_name(config, cmd_args, config_parser)
    # load best policy
    _, eval_path, _, _ = get_log_paths(config.logDir, run_name)
    print("Loading policy...")
    policy = MultiInputPolicy.load(os.path.join(eval_path, "best_policy"))

    # images
    bridge = CvBridge() 
    img_wh = 64
    min_max_hue_obj = [150,180]
    min_max_saturation_obj = [0,255] 
    min_max_val_obj = [0,255] 

    # load VAE
    print("Loading VAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.inference_mode():
        vae = VAE(latent_dim=config.latentDim)
        vae.load_state_dict(torch.load(os.path.join(config.logDir, "net", f"vae_latentdim_{config.latentDim}_imgHeight_{img_wh}_imgWidth_{img_wh}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{config.useSimConfig}")))
        vae.to(device)
    
    # ROS
    print("Init pubishers and subscribers...")
    rospy.init_node("eval_policy_node")
    # camera
    msg = rospy.wait_for_message("camera/rgb/image_raw", Image)
    usb_cam_sub = rospy.Subscriber("camera/rgb/image_raw", Image, callback=new_msg_callback)
    # controller
    controller = rospy.get_param("/controller")
    # init publisher
    panda_pose_pub = rospy.Publisher(f"{controller}/equilibrium_pose", PoseStamped, queue_size=1)
    # init subscriber
    franka_state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, callback=new_msg_callback)
    # get min ee z-pos
    min_ee_z_pos = rospy.get_param(f"{controller}/min_ee_z_pos") 

    print("Waiting for initial EE pose...")
    initial_ee_pos = wait_for_initial_pose()
    
    # move to start pos
    print("Moving to start pose...")
    start_pos = np.array([-1.29152186e-05, 4.12401994e-01, 5.90489624e-01]) # start real config
    initial_ee_zpos = start_pos[2]
    last_ee_pos = currentFrankaMsg_to_lastEEPos()
    rate = rospy.Rate(4) # Hz (250 ms)
    while any(np.abs(start_pos - last_ee_pos) > 1e-3):
        min_ee_z_pos = rospy.get_param(f"{controller}/min_ee_z_pos")
        rate.sleep()
        last_ee_pos = currentFrankaMsg_to_lastEEPos()
        pusblish_start_pos(start_pos, panda_pose_pub, min_ee_z_pos)
    pusblish_start_pos(last_ee_pos, panda_pose_pub, min_ee_z_pos)

    # goal image
    print("Choose an object goal pose...")
    continue_goal = input("Continue? (y)")
    while continue_goal != "y":
        continue_goal = input("Continue? (y) ")
    rospy.wait_for_message("camera/rgb/image_raw", Image)
    goal_binary_image, goal_rgb_image = cameraImage_to_binaryImage()
    goal_binary_image_t = torch.from_numpy(goal_binary_image.reshape((1,1,img_wh,img_wh))).to(device)
    with torch.inference_mode():
        goal_latent_vec = vae.encoder(goal_binary_image_t)[0,:].cpu().numpy()
    
    # start pose
    print("Choose an object start pose...")
    continue_start = input("Continue? (y)")
    while continue_start != "y":
        continue_start = input("Continue? (y) ")
    rospy.wait_for_message("camera/rgb/image_raw", Image)
    object_binary_image, object_rgb_image = cameraImage_to_binaryImage()
    
    continue_start = input("Continue? (y)")
    while continue_start != "y":
        continue_start = input("Continue? (y)")
    
    current_observation = OrderedDict([("achieved_goal", None), ("desired_goal", goal_latent_vec), ("observation", None)])
    counter_ep_steps = 0
    while not rospy.is_shutdown(): #and counter_ep_steps < 50:
        print(f"Episode: {counter_ep_steps}")
        # adapt observation space
        if config.useGRUFeatExtractor:
            for key in policy.observation_space.keys():
                if key != 'desired_goal' and (counter_ep_steps == 0 or policy.observation_space[key].shape[0] == counter_ep_steps):
                    policy.observation_space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(counter_ep_steps+1,policy.observation_space[key].shape[1]), dtype=np.float32)
        counter_ep_steps += 1

        object_binary_image, object_rgb_image = cameraImage_to_binaryImage()
        
        # latent state object      
        object_binary_image_t = torch.from_numpy(object_binary_image.reshape((1, 1, img_wh, img_wh)).astype(np.float32)).to(device)
        with torch.inference_mode():
            object_latent_vec = vae.encoder(object_binary_image_t)[0,:].cpu().numpy()

        # observation
        observation = currentFrankaMsg_to_lastEEPos()[:2]
        achieved_goal = object_latent_vec.copy()
        desired_goal = goal_latent_vec.copy()

        if config.useGRUFeatExtractor:
            # observation shape: (episode length, length of a single observation)
            if current_observation["observation"] is None:
                assert current_observation["achieved_goal"] is None
                current_observation["observation"] = observation.reshape((1,-1))
                current_observation["achieved_goal"] = achieved_goal.reshape((1,-1)) 
            else:
                current_observation["observation"] = np.concatenate((current_observation["observation"], observation.reshape((1,-1))), axis=0)
                current_observation["achieved_goal"] = np.concatenate((current_observation["achieved_goal"], achieved_goal.reshape((1,-1))), axis=0)
            obs = deepcopy(current_observation)
        elif config.numStackedObs is not None:
            single_obs_length = 2
            single_ag_length = 6
            if current_observation["observation"] is None:
                current_observation["observation"] = np.tile(observation, config.numStackedObs)
                current_observation["achieved_goal"] = np.tile(achieved_goal, config.numStackedObs)
            else:
                current_observation["observation"] = np.roll(current_observation["observation"], -single_obs_length)
                current_observation["achieved_goal"] = np.roll(current_observation["achieved_goal"], -single_ag_length)

                current_observation["observation"][-single_obs_length:] = observation
                current_observation["achieved_goal"][-single_ag_length:] = achieved_goal
            current_observation["desired_goal"] = desired_goal
            obs = deepcopy(current_observation)
        else:
            obs = OrderedDict([
                                ("observation", observation),
                                ("achieved_goal", achieved_goal),
                                ("desired_goal", desired_goal),
                            ])

        action = policy.predict(obs, deterministic=True)[0]
        if config.numSimSteps == -1:
            n_steps = int(action[-1])
        else:
            n_steps = config.numSimSteps # ms

        desired_pose = get_new_pose(action, initial_ee_zpos, min_ee_z_pos)
        start_time = rospy.Time.now() # s 
        now = start_time
        while int((now - start_time).to_nsec()*1e-6) < n_steps:
            now = rospy.Time.now()
            publish_new_pos(desired_pose, panda_pose_pub)
        if n_steps != int((now - start_time).to_nsec()*1e-6):
            print(f"{int((now - start_time).to_nsec()*1e-6)} ms, desired: {n_steps} ms")
       
    if counter_ep_steps == 50:
        print("Maximum number of episode timesteps (50) reached.")  
    
    action = np.zeros(2)
    while not rospy.is_shutdown():
        desired_pose = get_new_pose(action, initial_ee_zpos, min_ee_z_pos)
        publish_new_pos(desired_pose, panda_pose_pub)