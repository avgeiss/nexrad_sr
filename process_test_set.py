import data_manager as dm
from glob import glob
import numpy as np
from keras.models import load_model
from imageio import imsave
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1" #selects gpu

#get output for the dense unet at x8 SR
dm.files = glob('./data/composite_dbz_test/*.png')
inputsx4 = np.array(dm.load_comp_refs(len(dm.files),ndownsamples=4))[:,:,:,np.newaxis]
net = load_model('./data/models/dense_unet_64.hd5')
outputs = list(np.squeeze(net.predict(inputsx4,batch_size=10,verbose=True)))
output_dir = './data/output_composite_dbz/dense_unet_64/'
output_files = dm.files
for i in range(len(outputs)):
    print('Saving ' + str(i))
    output_file = output_dir + output_files[i][26:]
    image = outputs[i]
    image = np.uint8(np.floor(((image+1.0)/2.0)*255.0))
    imsave(output_file,image)

#get output for the dense unet at x4 SR
dm.files = glob('./data/composite_dbz_test/*.png')
dm.nfiles = len(dm.files)
inputsx4 = np.array(dm.load_comp_refs(len(dm.files),ndownsamples=3))[:,:,:,np.newaxis]
net = load_model('./data/models/dense_unet.hd5')
outputs = list(np.squeeze(net.predict(inputsx4,batch_size=10,verbose=True)))
output_dir = './data/output_composite_dbz/dense_unet/'
output_files = dm.files
for i in range(len(outputs)):
    print('Saving ' + str(i))
    output_file = output_dir + output_files[i][26:]
    image = outputs[i]
    image = np.uint8(np.floor(((image+1.0)/2.0)*255.0))
    imsave(output_file,image)