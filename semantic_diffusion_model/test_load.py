import torch as th
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import matplotlib.pyplot as plt

print('Loading')
synthesized_images = th.load('2023-10-24_21-05-18_syn.pt')
real_images = th.load('2023-10-24_21-05-18_real.pt')
print(synthesized_images.shape)
real_images = (real_images + 1) / 2.0
synthesized_images = synthesized_images[:600]
real_images = real_images[:600]
synth_im = synthesized_images.cpu().float().numpy()
real_im = real_images.cpu().float().numpy()
print(np.max(real_im[0]))

for i in range(100):
    plt.imsave('./results/2023-10-24_21-05-18/images/' + str(i) + '.png', real_im[i, 0, :, :], cmap=plt.cm.bone)
    plt.imsave('./results/2023-10-24_21-05-18/samples/' + str(i) + '.png', synth_im[i, 0, :, :], cmap=plt.cm.bone)

print(synthesized_images.shape)
# split synthesized_images into 10 subsets
synthesized_images_split = th.split(synthesized_images, 200, dim=0)
# split real_images into 10 subsets
real_images_split = th.split(real_images, 200, dim=0)

fid = FrechetInceptionDistance(normalize=True, feature=768) #768
#for i in range(len(synthesized_images)):                
fid.update(synthesized_images, real=False)
fid.update(real_images, real=True)

fid_score = fid.compute().item()
print('FID: ', fid_score)
print(len(real_images_split))

# calculate KID
kid = KernelInceptionDistance(normalize=True, subset_size=200, feature=2048, subsets=3)
for i in range(len(real_images_split)):
    kid.update(synthesized_images_split[i], real=False)
    kid.update(real_images_split[i], real=True)

kid_mean, kid_std = kid.compute()
print('KID: ', (kid_mean, kid_std))