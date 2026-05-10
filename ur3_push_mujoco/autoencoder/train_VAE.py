import os, torch
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from ur3_push_mujoco.autoencoder.VAE import VAE
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--latentDim", type=int, default=6, help="latent dim VAE")
parser.add_argument("--fixedObjectHeight", type=lambda x: None if x == 'None' else float(x), default=None, help="object height")
parser.add_argument("--useSimConfig", type=int, choices=[0, 1], default=1, help="use sim or real camera config?")
config = parser.parse_args()

data_path = os.getenv("PANDA_PUSH_DATAPATH")
torch.manual_seed(4242)

num_epochs = int(1e9)
num_train = 20000 # number of images used for training; num_test_images = num_images - num_train
num_stop = 100 # stop training on no model improvement after num_stop epochs
batch_size_train = 256
learning_rate = 1e-5
img_height = 64
img_width = 64
best_model_path = os.path.join(data_path, "net", f"vae_latentdim_{config.latentDim}_imgHeight_{img_height}_imgWidth_{img_width}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{int(config.useSimConfig)}")
num_example_img = 5 # plot img + reconstruction for num_example_img after training 

# dirs
if not os.path.isdir(data_path):
    os.mkdir(data_path)
if not os.path.isdir(os.path.join(data_path, "data")):
    os.mkdir(os.path.join(data_path, "data"))
if not os.path.isdir(os.path.join(data_path, "net")):
    os.mkdir(os.path.join(data_path, "net"))
if not os.path.isdir(os.path.join(data_path, "plots")):
    os.mkdir(os.path.join(data_path, "plots"))

# select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# load image data
image_file = os.path.join(data_path, "data", f"images_imgHeight_{img_height}_imgWidth_{img_width}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{config.useSimConfig}.npy")
with open(image_file, "rb") as f:
    binary_images = np.load(f)

train_images_np = binary_images[:num_train,:,:]
test_images_np = binary_images[num_train:,:,:]

train_images = torch.from_numpy(train_images_np.reshape((-1,1,img_height,img_width)).astype(np.float32)).to(device)
test_images = torch.from_numpy(test_images_np.reshape((-1,1,img_height,img_width)).astype(np.float32)).to(device)

# train and test data loader
# shuffle=True: reshuffle after each epoch
train_dataset = TensorDataset(train_images)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size_train, shuffle=True) 

# VAE
vae = VAE(latent_dim=config.latentDim).to(device)

# optimizer
optimizer = optim.Adam(params=vae.parameters(),lr=learning_rate)

# training - loss tracking
fig, axs = plt.subplots(3,1, figsize=(15,10))
plt.subplots_adjust(wspace=0.05, hspace=0.3)

num_train_batches = len(train_dataloader)

mean_total_train_loss = np.zeros(num_epochs)
mean_rec_train_loss = np.zeros(num_epochs)
mean_kl_train_loss = np.zeros(num_epochs)

min_total_train_loss = np.zeros(num_epochs)
min_rec_train_loss = np.zeros(num_epochs)
min_kl_train_loss = np.zeros(num_epochs)

mean_total_test_loss = np.zeros(num_epochs)
mean_rec_test_loss = np.zeros(num_epochs)
mean_kl_test_loss = np.zeros(num_epochs)

# training - stop on no model improvement
cnt_stop = 0
best_test_loss = np.inf
best_state_dict = None

for epoch in range(0,num_epochs):
    # training - loss tracking
    total_train_loss = np.zeros(num_train_batches)
    rec_train_loss = np.zeros(num_train_batches)
    kl_train_loss = np.zeros(num_train_batches)

    # train
    for i_train, minibatch_train in enumerate(train_dataloader):
        # forward pass
        total_loss_tr, rec_loss_tr, kl_loss_tr = vae.compute_loss(minibatch_train[0]) # compute total loss
        # backward pass
        optimizer.zero_grad() # set gradients to zero
        total_loss_tr.backward()
        optimizer.step()
        # remember loss
        total_train_loss[i_train] = total_loss_tr.detach().cpu().numpy()
        rec_train_loss[i_train] = rec_loss_tr.detach().cpu().numpy()
        kl_train_loss[i_train] = kl_loss_tr.detach().cpu().numpy()

    # test
    with torch.no_grad():
        # shuffle test images
        perm = torch.randperm(test_images.shape[0])
        test_images = test_images[perm]
        total_loss_te, rec_loss_te, kl_loss_te = vae.compute_loss(test_images)
    
    # track mean loss 
    mean_total_train_loss[epoch] = np.log(np.mean(total_train_loss)) 
    mean_rec_train_loss[epoch] = np.log(np.mean(rec_train_loss)) 
    mean_kl_train_loss[epoch] = np.log(np.mean(kl_train_loss)) 

    mean_total_test_loss[epoch] = np.log(total_loss_te.cpu().numpy())
    mean_rec_test_loss[epoch] = np.log(rec_loss_te.cpu().numpy())
    mean_kl_test_loss[epoch] = np.log(kl_loss_te.cpu().numpy())

    print(f"epoch {epoch}, stop counter: {cnt_stop}, mean total loss - train: {mean_total_train_loss[epoch]:.6f}, test: {mean_total_test_loss[epoch]:.6f}, mean rec loss - train: {mean_rec_train_loss[epoch]:.6f}, test: {mean_rec_test_loss[epoch]:.6f}, mean kl loss - train: {mean_kl_train_loss[epoch]:.6f}, test: {mean_kl_test_loss[epoch]:.6f}")

    # plot train and test loss
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    x = np.arange(0,epoch)
    
    axs[0].plot(x, mean_total_train_loss[:epoch], "-", color="blue", label="train")
    axs[0].plot(x, mean_total_test_loss[:epoch], "-", color="red", label="test")
    axs[0].set_ylabel("log loss")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title("Mean Total Loss per Epoch")

    axs[1].plot(x, mean_rec_train_loss[:epoch], "-", color="blue")
    axs[1].plot(x, mean_rec_test_loss[:epoch], "-", color="red")
    axs[1].set_ylabel("log loss")
    axs[1].grid()
    axs[1].set_title("Mean Reconstruction Loss per Epoch")

    axs[2].plot(x, mean_kl_train_loss[:epoch], "-", color="blue")
    axs[2].plot(x, mean_kl_test_loss[:epoch], "-", color="red")
    axs[2].set_xlabel("epoch")
    axs[2].set_ylabel("log loss")
    axs[2].grid()
    axs[2].set_title("Mean Similarity Loss per Epoch")

    # new best model?
    if mean_total_test_loss[epoch] < best_test_loss:
        best_test_loss = mean_total_test_loss[epoch]
        best_state_dict = deepcopy(vae.state_dict())
        cnt_stop = 0
    else:
        cnt_stop += 1

    if cnt_stop == num_stop:
        print(f"Training stopped after {epoch+1} epochs, because of no model improvement.")
        break
        
# save best model for inference
torch.save(best_state_dict, best_model_path)
# save plot
plt.savefig(os.path.join(data_path, "plots", f"loss_vae_latentdim_{config.latentDim}_imgHeight_{img_height}_imgWidth_{img_width}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{config.useSimConfig}.png"))

# load best model and plot some example images and their reconstruction
test_images_plot = test_images[:num_example_img].cpu()

with torch.inference_mode():
    vae_test = VAE(latent_dim=config.latentDim)
    vae_test.load_state_dict(torch.load(best_model_path))
    reconstructions = vae_test(test_images_plot)
    
# to numpy
test_images_plot.numpy()
reconstructions.numpy()

fig, axs = plt.subplots(2,num_example_img, figsize=(8,3))
plt.subplots_adjust(wspace=0.05, hspace=0.05)
for i in range(0, num_example_img):
    axs[0,i].imshow(test_images_plot[i,0], cmap="gray")
    axs[0,i].set_xticks([])
    axs[0,i].set_yticks([])

    axs[1,i].imshow(reconstructions[i,0], cmap="gray")
    axs[1,i].set_xticks([])
    axs[1,i].set_yticks([])
    
    if i==0:
        axs[0,i].set_ylabel("test image")
        axs[1,i].set_ylabel("reconstruction")

plt.savefig(os.path.join(data_path, "plots", f"reconstructions_vae_latentdim_{config.latentDim}_imgHeight_{img_height}_imgWidth_{img_width}_fixedObjectHeight_{config.fixedObjectHeight}_simConfig_{config.useSimConfig}.png"))
