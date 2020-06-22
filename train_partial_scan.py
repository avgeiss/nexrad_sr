#trains dense u-net on partial ppi scans
import data_manager as dm
import neural_networks as nets
import numpy as np
import keras.backend as K
from keras.models import load_model
from keras.optimizers import Adam
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1" #selects gpu

#constants:
MODEL_NAME = 'dense_unet_partial'
INPUT_SIZE = (48,48,1)    #(128,128,1) (64,64,1)
UPSAMPLES = 2
EPOCHS = 50
EPOCHS_LOW_LR = 20

#load in the radar data:
print('Loading Data')
full_scans = dm.load_comp_refs(ndownsamples=0)
n_scans = len(full_scans)

#split to training and validation sets and get subsamples of scans
print('Getting inputs and targets')
tidx, vidx = dm.validation_idx(0.25,n_scans,blocks=12,buf_size=24)
#tidx, vidx = dm.validation_idx(0.2,n_scans,blocks=2,buf_size=2)#to test that the training will run

#get neural network
print('Prepping Model')
model = getattr(nets, 'dense_unet')(INPUT_SIZE,upsamples=UPSAMPLES,base_channels=24)
model.compile(loss='MSE',metrics=['MAE'],optimizer=Adam(lr=0.0001))
model.summary()

#train the neural network:
benchmarks, mse, mae, min_mse = [], [], [], 10000.0  #placeholder
for i in range(EPOCHS + EPOCHS_LOW_LR):
    #get a new set of subsamples:
    inputs, targets = dm.get_partial_scan_dataset(full_scans)
    validation_inputs, validation_targets = inputs[vidx,:,:,:], targets[vidx,:,:,:]
    inputs, targets = inputs[tidx,:,:,:], targets[tidx,:,:,:]
    
    #compute benchmarks for the new subsamples:
    vbenchmarks = dm.compute_error_benchmarks(validation_targets,UPSAMPLES)
    print(vbenchmarks)
    benchmarks.append(vbenchmarks)
    
    #check to see if the learning rate needs to be reduced:
    if i == EPOCHS:
        print('Reducing learning rate...')
        model = load_model('./data/models/' + MODEL_NAME + '.hd5')
        K.set_value(model.optimizer.lr,0.1*K.eval(model.optimizer.lr))
        
    #train for an epoch:
    model.fit(inputs,targets,batch_size=5,epochs=1,verbose=1,shuffle=True)
    
    #test the model on the validation set:
    vout = model.predict(validation_inputs,batch_size=5,verbose=1)
    mse.append(np.mean((vout-validation_targets)**2))
    mae.append(np.mean(np.abs(vout-validation_targets)))
    
    #save model performance
    errors = np.stack((np.array(mse),np.array(mae)),axis=1)
    np.save('./data/models/' + MODEL_NAME + '_training_error.npy',errors)
    np.save('./data/models/' + MODEL_NAME + '_benchmark_error.npy',np.array(benchmarks))
    print(str(i) + ':    ' + str(mse[-1]) + '    ' + str(mae[-1]))
    
    #if the model performs best for this epoch save it:
    if mse[-1]<min_mse:
        print('Saving model...')
        model.save('./data/models/' + MODEL_NAME + '.hd5')
        min_mse = mse[-1]