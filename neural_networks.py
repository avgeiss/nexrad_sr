#definition of neural networks
from keras.models import Model
from keras.layers import Conv2D, Input, BatchNormalization, concatenate, UpSampling2D, MaxPooling2D, Activation
from keras.utils import plot_model
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1" #selects gpu

################    COMMON NEURAL NETWORK COMPONENTS    #######################

#A convolutional layer with batch normalization and relu included:
def conv(x,channels,filter_size=3):
    x = Conv2D(channels,(filter_size,filter_size),padding='same',activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def output_layer(x,filter_size=3):
    return Conv2D(1,(filter_size,filter_size),padding='same',activation='tanh')(x)

#upsampling and downsampling:
def upsample(x):
    x = UpSampling2D((2,2),interpolation='bilinear')(x)
    return x
def downsample(x):
    x = MaxPooling2D(pool_size=(2,2),padding='same')(x)
    return x

#Define some blocks of layers:
def res_block(x,channels,layers):
    xin = x
    for i in range(layers):
        x = conv(x,channels)
    x = concatenate([xin,x],axis=3)
    x = conv(x,channels,filter_size=5)
    return x

def dense_block(x,channels,layers):
    for i in range(layers):
        x_new = conv(x,channels)
        x = concatenate([x,x_new],axis=3)
    x = Conv2D(channels,(1,1),padding='same',activation='linear')(x)
    return x

def conv_block(x,channels,layers,filter_size=3):
    for i in range(layers):
        x = conv(x,channels,filter_size)
    return x

#recursively construct a u-net:
def unet(x,channels,rate,level,block_func,block_size):
    x = block_func(x,channels,block_size)
    if level>0:
        ux = downsample(x)
        ux = unet(ux,int(channels*rate),rate,level-1,block_func,block_size)
        ux = upsample(ux)
        x = concatenate([x,ux],axis=3)
        x = block_func(x,channels,block_size)
    return x

############################    MODEL DEFINITIONS    ##########################
#Definitions of models used in the paper:
def dense_unet(input_shape,upsamples=2,base_channels=12):
    #u-net module            
    x_in = Input(input_shape)
    x = unet(x_in,base_channels,2.1,3,dense_block,3)
    #upsampling module
    for i in range(upsamples):
        x = upsample(x)
        x = dense_block(x,int(base_channels/(i+1)),4)
    #return a model
    x_out = output_layer(x)
    return Model(inputs=x_in,outputs=x_out)

def dense_unet_64(input_shape,upsamples=3,base_channels=12):
    return dense_unet(input_shape,upsamples=upsamples,base_channels=base_channels)

def classic_unet(input_shape,upsamples=2,base_channels=40):
    #u-net module
    x_in = Input(input_shape)
    x = unet(x_in,base_channels,1.5,3,conv_block,2)
    #upsampling module
    for i in range(upsamples):
        x = upsample(x)
        x = dense_block(x,int(base_channels/(i+1)),3)
    #return a model
    x_out = output_layer(x)
    return Model(inputs=x_in,outputs=x_out)

def res_net(input_shape,upsamples=2,base_channels=80):
    x_in = Input(input_shape)
    x = res_block(x_in,base_channels,4)
    x = res_block(x,base_channels,4)
    for i in range(upsamples):
        x = upsample(x)
        for j in range(2):
            x = res_block(x,int(base_channels/(2**((i+1)*2))),4)
    x_out = output_layer(x)
    return Model(inputs=x_in,outputs=x_out)

def dense_net(input_shape,upsamples=2,base_channels=128):
    x_in = Input(input_shape)
    x = dense_block(x_in,base_channels,4)
    for i in range(upsamples):
        x = upsample(x)
        for j in range(2):
            x = dense_block(x,int(base_channels/(i+2)),4)
    x_out = output_layer(x)
    return Model(inputs=x_in,outputs=x_out)

def conv_net(input_shape,upsamples=2,base_channels=128):
    x_in = Input(input_shape)
    x = conv_block(x_in,base_channels,3,5)
    for i in range(upsamples):
        x = upsample(x)
        x = conv_block(x,int(base_channels/(2**(i*2+2))),3,5)
    x_out = output_layer(x)
    return Model(inputs=x_in,outputs=x_out)

def sr_net(input_shape,upsamples=2):
    x_in = Input(input_shape)
    x = upsample(upsample(x_in))
    x = conv(x,64,11)
    x = conv(x,32,7)
    x = conv(x,32,1)
    x_out = output_layer(x,5)
    return Model(inputs=x_in,outputs=x_out)
    

########################    TESTING CODE    ###################################
if __name__ == '__main__':
    model_names = ['conv_net','dense_unet','classic_unet','res_net','dense_net','conv_net','sr_net']
    for name in model_names:
        model = locals()[name]((128,128,1),upsamples=2)
        model.compile(loss='MSE',optimizer='adam',metrics=['MAE'])
        text = model.summary()
        plot_model(model,show_layer_names=False,show_shapes=True,to_file='./data/model_plots/' + name + '_plot.png')