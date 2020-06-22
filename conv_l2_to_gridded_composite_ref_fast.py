#reads in NEXRAD L2 data and converts to composite reflectivity on a cartesian
#grid
import pyart as art
import numpy as np
from glob import glob
from imageio import imsave
from multiprocessing import Pool

#CONSTANTS:
#range of grid in meters:
cart_range = [-400000,400000]
#grid resolution:
cart_res = 1024
#location of the L2 data files
l2_data_path = './data/l2/*'
output_path = './data/composite_dbz/'
#whether to use parallel processing:
NTHREADS = 8
#the maximum/minimum reflectivity measured over all scan levels:
max_ref = 94.5
min_ref = -32.0

#list of all the file names:
files = glob(l2_data_path)
files.sort()

#remove files that were already processed:
print('Processing ' + str(len(files)) + ' files...')

#takes a LII sweep as input and performs NN interpolation for a particular
#field on to a Cartesian grid
def nearest_neighbor_interpolator(dbz,azm,rng,crng=cart_range,res=cart_res):
    #create the cartesian target grid
    x = np.linspace(crng[0],crng[1],res)
    x,y  = np.meshgrid(x,x)
    #polar coords associated with the cartesian gridpoints
    theta = 360.0*(np.arctan2(y,x)+np.pi)/(2*np.pi)
    r = np.sqrt(x**2+y**2)
    #do the nearest neighbor interpolation
    gridded = np.zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            azmidx = np.argmin(np.abs(azm-theta[i,j]))
            rngidx = np.argmin(np.abs(rng-r[i,j]))
            gridded[i,j] = dbz[azmidx,rngidx]
    gridded = np.double(gridded)
    return gridded
    
#function to extract a volume scan and compute composite reflectivity
def get_composite_ref(fname):
    #read in some metadata
    l2 = art.io.nexrad_level2.NEXRADLevel2File(fname)
    #ensure that the radar is using VCP12/212/215 scan pattern (typical scan pattern
    #when there is precip in a significant portion of the domain)
    vcp = l2.get_vcp_pattern()
    if vcp != 12 and vcp != 212 and vcp != 215:
        print('vcp = ' + str(vcp) + ' skipping...')
        return False
    #load in the volume scan:
    arch = art.io.nexrad_archive.read_nexrad_archive(fname)
    #read each sweep and align azimuths:
    sweep_ref = []
    for i in range(min(6,arch.nsweeps)): #only use the first 6 scans
        sweep = arch.extract_sweeps([i])
        ref = sweep.fields['reflectivity']['data'].data
        azm = sweep.azimuth['data']
        daz = np.argmin(azm)
        azm = np.roll(azm,-daz,axis=0)
        if i == 0:
            AZM = azm
        ref = np.roll(ref,-daz,axis=0)
        ref = np.double(ref)
        if len(azm)==720: #only use the high resolution scans
            sweep_ref.append(ref)
    composite = np.max(np.array(sweep_ref),axis=0)
    rng = sweep.range['data']
    composite = nearest_neighbor_interpolator(composite,AZM,rng)
    #convert to uint8 format:
    composite = 255.0*(composite-min_ref)/(max_ref-min_ref)
    composite[composite<0] = 0
    composite[composite>255] = 255
    composite = np.uint8(composite)
    #save to the output directory:
    output_fname = '' + output_path + fname[-19:-4] + '.png'
    imsave(output_fname,composite)
    return composite
    
def helper(fnum):
    try:
        fname = files[fnum]
        print('Processing: ' + fname)
        get_composite_ref(fname)
    except:
        print(fname + 'FAILED!')

#process the files:
if NTHREADS>1:
    p = Pool(NTHREADS)
    p.map(helper,range(len(files)),chunksize=1)
else:
    for f in range(len(files)):
        helper(f)