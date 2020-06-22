#loads and processes NEXRAD dataset
from glob import glob
from imageio import imread
import numpy as np
from multiprocessing import Pool
from functools import partial
from os import path
from PIL import Image

#CONSTANTS:
#the range of reflectivity values:
max_ref = 94.5
min_ref = -32.0
#parallel processing:
THREADS = 8

#########################    RESIZING SCHEMES    ##############################
#halves scan resolution n-times
def downsampler(image,n=1):
    image = np.copy(image)
    pow2 = 2**n
    image.shape = [int(image.shape[0]/pow2),pow2,int(image.shape[1]/pow2),pow2]
    image = np.mean(image,axis=(1,3)).squeeze()
    return image
        
#resizes a scan using requested interpolation scheme
def resize(scan,imsize,scheme):
    #convert scan to an image so I can use PIL resizing tool:
    scan = np.uint8(((np.copy(scan)+1.0)/2.0)*255)
    scan = np.stack((scan,scan,scan),axis=2)
    #resize with PIL
    scan = Image.fromarray(scan).resize((imsize,imsize),scheme)
    #convert back to matrix
    scan = np.array(scan,dtype='float16')
    scan = 2.0*(scan[:,:,0].squeeze()/255.0)-1.0
    return scan

###############################    LOAD DATA:    ##############################
files = glob('./data/composite_dbz/*.png')
files = glob('./data/composite_dbz_test/*.png')
nfiles = len(files)

#a function to read in composite reflectivities by file index:
def ref_reader(file_number,ndownsamples=1):
    dbz = imread(files[file_number])
    dbz = 2.0*(np.float16(dbz)/255.0)-1.0
    if ndownsamples>0:
        dbz = downsampler(dbz,n=ndownsamples)
    return dbz

#reads in all composite reflectivity files:
def load_comp_refs(n=nfiles,ndownsamples=1):
    p = Pool(THREADS)
    ref = p.map(partial(ref_reader,ndownsamples=ndownsamples),range(n))
    p.close();p.join()
    return ref

###################    DATA AUGMENTATION    ###################################
def augment(inp,targ):
    if np.round(np.random.uniform(0.0,1.0)) == 1.0:
        inp = np.flip(inp,axis=0)
        targ = np.flip(targ,axis=0)
    if np.round(np.random.uniform(0.0,1.0)) == 1.0:
        inp = np.flip(inp,axis=1)
        targ = np.flip(targ,axis=1)
    n_rots = np.int(np.round(np.random.uniform(0.0,3.0)))
    inp = np.rot90(inp,n_rots)
    targ = np.rot90(targ,n_rots)
    return inp,targ

def augment_training_set(inps,targs):
    for i in range(inps.shape[0]):
        inps[i,:,:,:],targs[i,:,:,:] = augment(inps[i,:,:,:],targs[i,:,:,:])
    return inps,targs

###################    TRAINING AND VALIDATION SET CREATION    ################
#generates indices for a training and validation set
def validation_idx(split=0.25,n=100,blocks=2,buf_size=5):
    #want to be consistent between training runs, if an identical validation 
    #split has been generated before load it:
    dirname = './data/validation_splits/'
    fname =  dirname + 'vsplit_' + str(split) + '_' + str(n) + '_' + str(blocks) + '_' + str(buf_size) + '.npz'
    if path.exists(fname):
        vsplit = np.load(fname)
        tidx = vsplit['tidx']
        vidx = vsplit['vidx']
    else:
        #break the indices into blocks, one block for each period to take samples
        #for the validation set from:
        break_size = int(n/blocks)
        idx = np.array(range(0,break_size*blocks))
        idx.shape = (blocks,break_size)
        idx = idx.tolist()
        #extract a contiguous set of indices from each block for the validation set
        vblock_size = int(n*split/blocks+2*buf_size)
        vidx = [];tidx = []
        for b in idx:
            subsample_start = np.random.randint(0,break_size-vblock_size)
            subsample = b[subsample_start:(subsample_start+vblock_size)]
            tidx.append(np.delete(b,range(subsample_start,subsample_start+vblock_size)))
            vidx.append(subsample[buf_size:-buf_size])
        vidx = np.array(vidx).flatten()
        tidx = np.array(tidx).flatten()
        np.savez(fname,tidx=tidx,vidx=vidx)
    return tidx, vidx

def get_full_scan_dataset(targets,n_upsamples=2):
    p = Pool(THREADS)
    inputs = np.array(p.map(partial(downsampler,n=n_upsamples),targets))
    targets = np.array(targets)
    targets = targets[:,:,:,np.newaxis]
    inputs = inputs[:,:,:,np.newaxis]
    p.close();p.join()
    return inputs,targets

def subsample_scan(ref,mnsz=192,mxsz=512):
    #first get a random scale and location:
    np.random.seed()
    scale = np.random.randint(mnsz,mxsz)
    scan_size = ref.shape[0]
    v_offset = np.random.randint(0,scan_size-scale)
    h_offset = np.random.randint(0,scan_size-scale)
    #get sample:
    sample = ref[v_offset:v_offset+scale,h_offset:h_offset+scale]
    sample = resize(sample,mnsz,Image.BILINEAR)
    #do random flips and rotations:
    sample,_ = augment(sample,sample)
    return sample

#generates a new training set of randomly sampled, scaled, flipped, and rotated
#samples from ppi scans
def get_partial_scan_dataset(refs,target_min_size=192,target_max_size=512,n_upsamples=2):
    p = Pool(THREADS)
    tar = p.map(partial(subsample_scan,mnsz=target_min_size,mxsz=target_max_size),refs)
    inp = p.map(partial(downsampler,n=n_upsamples),tar)
    inp = np.array(inp);tar = np.array(tar)
    inp = inp[:,:,:,np.newaxis];tar = tar[:,:,:,np.newaxis]
    p.close();p.join()
    return inp, tar

#############################    COMPUTE BENCHMARKS   #########################
def benchmark_error(im,n_downsamples=2):
    sz = im.shape
    schemes = [Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.LANCZOS]
    mse = [];  mae = []
    downsampled = downsampler(im,n_downsamples)#change this to get different input resolutions    
    for scheme in schemes:
        upsampled = resize(downsampled,sz[0],scheme)
        mse.append(np.mean((im-upsampled)**2.0))
        mae.append(np.mean(np.abs(im-upsampled)))
    errors = np.stack((np.array(mse),np.array(mae)),axis=1)
    return errors

def compute_error_benchmarks(scans,n_downsamples):
    p = Pool(THREADS)
    errors = p.map(partial(benchmark_error,n_downsamples=n_downsamples),list(scans.squeeze()))
    errors = np.mean(np.array(errors),axis=0)
    p.close();p.join()
    return errors

#####################    TESTING CODE    ######################################
if __name__ == '__main__':
    refs = load_comp_refs()
    inputs, targets = get_full_scan_dataset(refs,2)
    tidx, vidx = validation_idx(0.2,100,4,2)