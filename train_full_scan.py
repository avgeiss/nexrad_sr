#trains dense u-net on full ppi scans
import data_manager as dm
import neural_networks as nets
import numpy as np
import keras.backend as K
from keras.models import load_model
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1" #selects gpu
MODEL_NAME = 'dense_unet'   #'dense_unet' 'dense_unet_64' 'classic_unet' 'res_net' 'dense_net' 'conv_net' 'sr_net'
INPUT_SIZE = (128,128,1)    #(128,128,1) (64,64,1)
EPOCHS = 90
EPOCHS_LOW_LR = 10

#get training and validation data
print('Loading Data')
targets = dm.load_comp_refs(ndownsamples=1)
n_scans = len(targets)
tidx, vidx = dm.validation_idx(0.25,n_scans,blocks=12,buf_size=24)
#tidx, vidx = dm.validation_idx(0.2,n_scans,blocks=2,buf_size=2)#to test that the training will run
print('Getting inputs and targets')
upsamples = int(np.log2(512)-np.log2(INPUT_SIZE[0]))
inputs, targets = dm.get_full_scan_dataset(targets,n_upsamples=upsamples)
validation_inputs, validation_targets = inputs[vidx,:,:,:], targets[vidx,:,:,:]
inputs, targets = inputs[tidx,:,:,:], targets[tidx,:,:,:]

#get neural network
print('Prepping Model')
model = getattr(nets, MODEL_NAME)(INPUT_SIZE,upsamples=upsamples)
model.compile(loss='MSE',optimizer='adam',metrics=['MAE'])
model.summary()

#print benchmark scores for comparison:
print('Computing Benchmarks');
vbenchmarks = dm.compute_error_benchmarks(validation_targets,upsamples)
print(vbenchmarks)

#train the neural network:
mse, mae, min_mse = [], [], 10000.0  #placeholder
for i in range(EPOCHS + EPOCHS_LOW_LR):
    if i == EPOCHS:
        #reduce the learning rate
        print('Reducing learning rate...')
        model = load_model('./data/models/' + MODEL_NAME + '.hd5')
        K.set_value(model.optimizer.lr,0.1*K.eval(model.optimizer.lr))
    #train for an epoch:
    inputs,targets = dm.augment_training_set(inputs,targets)
    model.fit(inputs,targets,batch_size=5,epochs=1,verbose=1,shuffle=True)
    #test the model on the validation set:
    vout = model.predict(validation_inputs,batch_size=5,verbose=1)
    mse.append(np.mean((vout-validation_targets)**2))
    mae.append(np.mean(np.abs(vout-validation_targets)))
    errors = np.stack((np.array(mse),np.array(mae)),axis=1)
    np.save('./data/models/' + MODEL_NAME + '_training_error.npy',errors)
    print(str(i) + ':    ' + str(mse[-1]) + '    ' + str(mae[-1]))
    #if the model performs best for this epoch save it:
    if mse[-1]<min_mse:
        print('Saving model...')
        model.save('./data/models/' + MODEL_NAME + '.hd5')
        min_mse = mse[-1]