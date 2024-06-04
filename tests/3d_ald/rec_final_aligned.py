#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import dxchange
import holotomo
import matplotlib.pyplot as plt
import cupy as cp
import scipy.ndimage as ndimage
import numpy as np
import sys

# get_ipython().run_line_magic('matplotlib', 'inline')

# !jupyter nbconvert --to script rec_final.ipynb


# # Init data sizes and parametes of the PXM of ID16A

# In[26]:


cp.cuda.Device(int(sys.argv[1])).use()
ntheta = int(sys.argv[2])#750  # number of angles (rotations)
ptheta = int(sys.argv[3])  # holography chunk size for GPU processing
binning = int(sys.argv[4])
niter = int(sys.argv[5])
iter_step = int(sys.argv[6])
ndist = int(sys.argv[7])
st =  int(sys.argv[8])
same_probe =  sys.argv[9]=='True'
flg_show = False

# cp.cuda.Device(0).use()
# ntheta = 10  # number of angles (rotations)
# ptheta = 5  # holography chunk size for GPU processing
# binning = 0
# niter = 33
# iter_step = 8
# ndist = 4
# st = 0
# same_probe = True
# flg_show = True

cropx_left = 256+64
cropx_right = 256-64
cropy_up = 0
cropy_down = 256+128+32+16+64-16-32+64

n = (2048-cropx_left-cropx_right)//2**binning

ne = (2048+256+128+96)//2**binning
pn = 32  # tomography chunk size for GPU processing
center = n/2  # rotation axis

# ID16a setup
detector_pixelsize = 3e-6
energy = 17.05 #[keV] xray energy
focusToDetectorDistance = 1.208 # [m]
sx0 = -2.493e-3
z1 = np.array([1.5335e-3,1.7065e-3,2.3975e-3,3.8320e-3])[:ndist]-sx0
z2 = focusToDetectorDistance-z1
distances = (z1*z2)/focusToDetectorDistance
magnifications = focusToDetectorDistance/z1
voxelsize = detector_pixelsize/magnifications[0]*2**binning  # object voxel size
norm_magnifications = magnifications/magnifications[0]
distances = distances*norm_magnifications**2

z1p = z1[0]  # positions of the probe for reconstruction
z2p = z1-np.tile(z1p, len(z1))
magnifications2 = (z1p+z2p)/z1p
distances2 = (z1p*z2p)/(z1p+z2p)
norm_magnifications2 = magnifications2/(z1p/z1[0])  # 
distances2 = distances2*norm_magnifications2**2
distances2 = distances2*(z1p/z1)**2
print(norm_magnifications*ne/n)


# In[27]:


def remove_outliers(data):
    """Remove outliers"""
    r = 5
    if len(data.shape)==4:
        fdata = ndimage.median_filter(data, [1,1, r, r])
    if len(data.shape)==3:
        fdata = ndimage.median_filter(data, [1, r, r])
    elif len(data.shape)==2:
        fdata = ndimage.median_filter(data, [r, r])
    ids = np.where(np.abs(fdata-data) > 0.04*np.abs(fdata))
    print(len(ids))
    data[ids] = fdata[ids]
    return data


# ## Read data

# In[28]:


data00 = np.zeros([ndist,ntheta,n,n],dtype='float32')
ref00 = np.zeros([ndist,n,n],dtype='float32')
ref01 = np.zeros([ndist,n,n],dtype='float32')
dark00 = np.zeros([ndist,n,n],dtype='float32')

mmeans = np.zeros(8)

for k in range(ndist):
    for j in range(0,ntheta):
        jtheta=(st+j*1500//ntheta)
        # print(jtheta)
        fname = f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_{k+1}_/3d_ald4_ht_10nm_{k+1}_{jtheta:04}.edf'
        tmp = dxchange.read_edf(fname)[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
        for kb in range(binning):
            tmp = (tmp[::2]+tmp[1::2])/2
            tmp = (tmp[:,::2]+tmp[:,1::2])/2
        data00[k,j] = tmp

    tmp = dxchange.read_edf(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_{k+1}_/refHST0000.edf')[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
    
    for kb in range(binning):
        tmp = (tmp[::2]+tmp[1::2])/2
        tmp = (tmp[:,::2]+tmp[:,1::2])/2
    ref00[k] = tmp

    tmp = dxchange.read_edf(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_{k+1}_/refHST1500.edf')[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
    for kb in range(binning):
        tmp = (tmp[::2]+tmp[1::2])/2
        tmp = (tmp[:,::2]+tmp[:,1::2])/2
    ref01[k] = tmp

    tmp = dxchange.read_edf(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_{k+1}_/dark.edf')[0,cropy_up:2048-cropy_down,cropx_left:2048-cropx_right]
    for kb in range(binning):
        tmp = (tmp[::2]+tmp[1::2])/2
        tmp = (tmp[:,::2]+tmp[:,1::2])/2
    dark00[k] = tmp
if flg_show:
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    im=axs[0].imshow(data00[0,0],cmap='gray',vmax=10000)
    fig.colorbar(im)
    im=axs[1].imshow(ref00[0],cmap='gray',vmax=10000)
    fig.colorbar(im)
    im=axs[2].imshow(data00[0,0]/ref00[0],cmap='gray')#,vmax=0.9,vmin=0.2)
    fig.colorbar(im)

# data00 = remove_outliers(data00)
# ref00 = remove_outliers(ref00)
# ref01 = remove_outliers(ref01)
# dark00 = remove_outliers(dark00)
# plt.imshow(dark00)
# plt.show()

data00 -= dark00[:,np.newaxis]
ref00 -= dark00
ref01 -= dark00

if flg_show:
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    im=axs[0].imshow(data00[0,0],cmap='gray',vmax=10000)
    fig.colorbar(im)
    im=axs[1].imshow(ref00[0],cmap='gray',vmax=10000)
    fig.colorbar(im)
    im=axs[2].imshow(data00[0,0]/ref00[0],cmap='gray')
    fig.colorbar(im)
# for k in range(ndist):
#     plt.plot(np.mean(data00[k],axis=(1,2)))
#     plt.show()


# 

# In[29]:


for k in range(ndist):
    v = np.linspace(np.mean(ref00[k]),np.mean(ref01[k]),1500)[st:1500:1500//ntheta]/np.mean(ref00[k])
    data00[k]/=v[:,np.newaxis,np.newaxis]
    data00[k]*=np.mean(ref00[0])/np.mean(ref00[k])
for k in range(ndist):
    ref00[k]*=np.mean(ref00[0])/np.mean(ref00[k])


if flg_show:
    for k in range(ndist):
        plt.plot(np.mean(data00[k],axis=(1,2)),label=f'{k}')
    plt.legend()
    plt.show()
    
for k in range(ndist):
    data00[k]*=(np.mean(data00[0],axis=(1,2))/np.mean(data00[k],axis=(1,2)))[:,np.newaxis,np.newaxis]

if flg_show:
    for k in range(ndist):
        plt.plot(np.mean(data00[k],axis=(1,2)),label=f'{k}')
    plt.legend()
    plt.show()

for k in range(ndist):
    data00*=(np.mean(data00[0,0])/np.mean(data00,axis=(2,3)))[:,:,np.newaxis,np.newaxis]


if flg_show:
    for k in range(ndist):
        plt.plot(np.mean(data00[k],axis=(1,2)),label=f'{k}')
    plt.legend()
    plt.show()


# In[30]:


def apply_shift(psi, p):
    """Apply shift for all projections."""
    psi = cp.array(psi)
    p = cp.array(p)
    tmp = cp.pad(psi,((0,0),(n//2,n//2),(n//2,n//2)), 'symmetric')
    [x, y] = cp.meshgrid(cp.fft.rfftfreq(2*n),
                         cp.fft.fftfreq(2*n))
    shift = cp.exp(-2*cp.pi*1j *
                   (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res0 = cp.fft.irfft2(shift*cp.fft.rfft2(tmp))
    res = res0[:, n//2:3*n//2, n//2:3*n//2].get()
    return res


# already calculated
shifts_probe_nil = np.load('shifts_probe_nil.npy')[:,np.newaxis]/2**binning


#set probe at 2 as ref
shifts_probe_nil-=shifts_probe_nil[1]



shifts_probe = np.load('shifts_probe.npy')[:,st:1500:1500//ntheta]/2**binning
# sum
shifts_probe+=shifts_probe_nil


# # Dark-flat field correction

# In[31]:


data0 = data00.copy()
ref0 = ref00.copy()
ref0[ref0<0] = 0
data0[data0<0]=0

# data0 = remove_outliers(data0)
# ref0 = remove_outliers(ref0)

ref0_shifted_nil_check = apply_shift(ref0,-shifts_probe_nil[:,0])
ref0_shifted = np.tile(ref0[1:2,np.newaxis],[ndist,ntheta,1,1])
for k in range(ntheta):
    ref0_shifted[:,k] = apply_shift(ref0_shifted[:,k],shifts_probe[:,k])

rdata=data0/(ref0_shifted+1e-9)
# rdata=data0/(ref0[:,np.newaxis]+1e-9)
for k in range(ndist):
    dxchange.write_tiff(rdata[k],f'/data/viktor/tmp/tn{k}.tiff',overwrite=True)


# In[33]:


if flg_show:

    for k in range(ndist):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        im=axs[0].imshow(ref0[1]-ref0[k],cmap='gray',vmin=-200,vmax=200)
        axs[0].set_title('ref[0] -ref[k] ')
        fig.colorbar(im)
        im=axs[1].imshow(ref0_shifted_nil_check[1]-ref0_shifted_nil_check[k],cmap='gray',vmin=-200,vmax=200)
        axs[1].set_title('ref[0] -ref[k] shifted ')
        fig.colorbar(im)


# In[34]:


if flg_show:
    for k in range(ndist):
        fig, axs = plt.subplots(1, 2, figsize=(9, 4))
        im=axs[0].imshow(data00[k,-1],cmap='gray',vmin = 0,vmax=10000 )
        axs[0].set_title(f'rdata for theta {ntheta-1} dist {k}')
        fig.colorbar(im)
        im=axs[1].imshow(rdata[k,-1],cmap='gray')#,vmin = 0.4,vmax=.7)
        axs[1].set_title(f'rdata for theta {ntheta-1} dist {k}')
        fig.colorbar(im)


# In[10]:


rdata_scaled = rdata.copy()
for j in range(ntheta):
    for k in range(ndist):    
        a = ndimage.zoom(rdata[k,j],1/norm_magnifications[k])
        rdata_scaled[k,j] = a[a.shape[0]//2-n//2:a.shape[0]//2+n//2,a.shape[1]//2-n//2:a.shape[1]//2+n//2]

if flg_show:
    for k in range(ndist):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im=axs[0].imshow(rdata_scaled[0,0],cmap='gray')#,vmin = 0.4,vmax=0.7 )
        axs[0].set_title(f'shifted rdata_new_scaled for theta 0 dist {k}')
        fig.colorbar(im)
        im=axs[1].imshow(rdata_scaled[k,0],cmap='gray')#,vmin = 0.4,vmax=0.7 )
        axs[1].set_title(f'shifted rdata_new_scaled for theta {ntheta-1} dist {k}')
        fig.colorbar(im)        
        im=axs[2].imshow(rdata_scaled[0,0]-rdata_scaled[k,0],cmap='gray')#,vmin =-0.1,vmax=.1 )
        axs[2].set_title(f'difference')
        fig.colorbar(im)        


# In[11]:


shifts_random = np.zeros([ndist,ntheta,2],dtype='float32')
for k in range(ndist):
    s = np.loadtxt(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_{k+1}_/correct.txt').astype('float32')[st:1500:1500//ntheta]/norm_magnifications[k]    
    shifts_random[k,:,0] = s[:,1]+(1024-(2048+cropy_up-cropy_down)/2)*(1/norm_magnifications[k]-1)#/norm_magnifications[k]
    shifts_random[k,:,1] = s[:,0]+(1024-(2048+cropx_left-cropx_right)/2)*(1/norm_magnifications[k]-1)#/norm_magnifications[k]

shifts_correct3D =np.zeros([ntheta,2],dtype='float32')
s = np.loadtxt(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_/correct_correct3D.txt').astype('float32')[st:1500:1500//ntheta]
shifts_correct3D[:,0] = s[:,1]
shifts_correct3D[:,1] = s[:,0]

shifts_random+=shifts_correct3D

if flg_show:
    plt.plot(shifts_random[0,:,1])
    plt.show()


# # Total shifts in pixels before normalized scaling

# In[12]:


import scipy.io 
shifts_new = -scipy.io.loadmat('/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_/rhapp_py.mat')['rhapp'][:,:,st:1500:1500//ntheta].swapaxes(0,2).swapaxes(0,1)[:ndist]
shifts_new+=shifts_random
shifts_new/=2**binning
shifts = shifts_new



# # TEST: Scale initial data to 1 magnification and apply all shifts

# In[13]:


rdata_scaled_new = np.ones([ndist,ntheta,ne,ne],dtype='float32')#rdata.copy()
mask = np.zeros([ne,ne],dtype='float32')#rdata.copy()
for j in range(ntheta):
    for k in range(ndist):    
        a = apply_shift(rdata[k,j:j+1],-shifts[k,j:j+1]*norm_magnifications[k,np.newaxis,np.newaxis])[0]# note first shift then magnification
        a = ndimage.zoom(a,1/norm_magnifications[k])
        if a.shape[-1]%2==1:
            a=a[:-1,:-1]
        if j==0:
            mask[-a.shape[0]//2+ne//2:a.shape[0]//2+ne//2,-a.shape[0]//2+ne//2:a.shape[0]//2+ne//2]+=1
        a = np.pad(a,((-a.shape[0]//2+ne//2,-a.shape[0]//2+ne//2),(-a.shape[0]//2+ne//2,-a.shape[0]//2+ne//2)),'symmetric')        
        rdata_scaled_new[k,j] = a

if flg_show:
    for k in range(ndist):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        im=axs[0].imshow(rdata_scaled_new[0,-1,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],cmap='gray')#,vmin = 0.4,vmax=0.7)
        axs[0].set_title(f'shifted rdata_new_scaled for theta 0 dist {k}')
        fig.colorbar(im)
        im=axs[1].imshow(rdata_scaled_new[k,-1,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],cmap='gray')#,vmin = 0.4,vmax=0.7)
        axs[1].set_title(f'shifted rdata_new_scaled for theta {ntheta-1} dist {k}')
        fig.colorbar(im)        
        im=axs[2].imshow(rdata_scaled_new[0,-1,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2]-rdata_scaled_new[k,-1,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2],cmap='gray')#,vmin = -0.1,vmax=0.1)
        axs[2].set_title(f'difference')
        fig.colorbar(im)          
        
    


# In[14]:


def CTFPurePhase(rads, wlen, dists, fx, fy, alpha):
   """
   weak phase approximation from Cloetens et al. 2002




   Parameters
   ----------
   rad : 2D-array
       projection.
   wlen : float
       X-ray wavelentgth assumes monochromatic source.
   dist : float
       Object to detector distance (propagation distance) in mm.
   fx, fy : ndarray
       Fourier conjugate / spatial frequency coordinates of x and y.
   alpha : float
       regularization factor.
       
   Return
   ------
   phase retrieved projection in real space
   """    
   numerator = 0
   denominator = 0    
   for j in range(0, len(dists)):    
       radsj = cp.array(rads[j])
       rad_freq = cp.fft.fft2(radsj)
       taylorExp = cp.sin(cp.pi*wlen*dists[j]*(fx**2+fy**2)) 
       numerator = numerator + taylorExp * (rad_freq)
       denominator = denominator + 2*taylorExp**2 
   numerator = numerator / len(dists)
   denominator = (denominator / len(dists)) + alpha
   phase = cp.real(  cp.fft.ifft2(numerator / denominator) )
   phase = 0.5 * phase
   return phase.get()

def multiPaganin(rads, wlen, dists, delta_beta, fx, fy, alpha):
    """
    Phase retrieval method based on Contrast Transfer Function. This 
    method relies on linearization of the direct problem, based  on  the
    first  order  Taylor expansion of the transmittance function.
    Found in Yu et al. 2018 and adapted from Cloetens et al. 1999


    Parameters
    ----------
    rad : 2D-array
        projection.
    wlen : float
        X-ray wavelentgth assumes monochromatic source.
    dist : float
        Object to detector distance (propagation distance) in mm.
    delta : float    
        refractive index decrement
    beta : float    
        absorption index
    fx, fy : ndarray
        Fourier conjugate / spatial frequency coordinates of x and y.
    alpha : float
        regularization factor.
        
    Return
    ------

    phase retrieved projection in real space
    """    
    numerator = 0
    denominator = 0    
    for j in range(0, len(dists)):    
        radsj = cp.array(rads[j])
        rad_freq = cp.fft.fft2(radsj)    
        taylorExp = 1 + wlen * dists[j] * np.pi * (delta_beta) * (fx**2+fy**2)
        numerator = numerator + taylorExp * (rad_freq)
        denominator = denominator + taylorExp**2 

    numerator = numerator / len(dists)
    denominator = (denominator / len(dists)) + alpha

    phase = cp.log(cp.real(  cp.fft.ifft2(numerator / denominator) ))    
    phase = (delta_beta) * 0.5 * phase

    
    return phase.get()

distances_rec = (distances/norm_magnifications**2)[:ndist]
fx = cp.fft.fftfreq(ne,d=voxelsize)
[fx,fy] = np.meshgrid(fx,fx)

wlen = 1.24e-9/energy
recMultiPaganin = np.zeros([ntheta,ne,ne],dtype='complex64')
recCTF = np.zeros([ntheta,ne,ne],dtype='complex64')
for k in range(ntheta):
    rads = rdata_scaled_new[:ndist,k]
    recMultiPaganin[k] = np.exp(1j*multiPaganin(rads, wlen, distances_rec,50, fx, fy, 1e-12))
    recCTF[k] = np.exp(1j*CTFPurePhase(rads, wlen, distances_rec,fx, fy, 1e-3))#[n//2:-n//2,n//2:-n//2]
if flg_show:
    plt.imshow(np.angle(recMultiPaganin[0]),cmap='gray')
    plt.colorbar()
    plt.show()
if flg_show:
    plt.imshow(np.angle(recCTF[0]),cmap='gray')
    plt.colorbar()
    plt.show()
dxchange.write_tiff(np.angle(recMultiPaganin),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{n}_{ntheta}_{ndist}_{st}/MultiPaganin.tiff',overwrite=True)
dxchange.write_tiff(np.angle(recCTF),         f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{n}_{ntheta}_{ndist}_{st}/recCTF.tiff',overwrite=True)




# 

# ## Create a solver class for holography

# In[15]:


pslv = holotomo.SolverHolo(ntheta, n, ne, ptheta, voxelsize, energy, distances, norm_magnifications,distances2,same_probe=same_probe)


# note ptheta parameter is used to define the number of angles for simultaneous processing by 1 gpu. Currently pntheta=ntheta. If the gpu does not have enough memory then ptheta should  be decreased
# norm_magnifications are magnifications for different distances, normalized by the magnification for the first distance

# ## Adjoint test

# In[16]:


data = data0.copy()
ref = ref0.copy()
arr1 = np.pad(np.array(data[0]+1j*data[0]).astype('complex64'),((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'symmetric')
prb1 = np.array(ref[:]+1j*ref[:]).astype('complex64')
print(arr1.shape,prb1.shape,shifts_probe.shape)
arr2 = pslv.fwd_holo_batch(arr1,prb1, shifts,shifts_probe=shifts_probe)
arr3 = pslv.adj_holo_batch(arr2,prb1, shifts,shifts_probe=shifts_probe)
prb3 = pslv.adj_holo_prb_batch(arr2,arr1, shifts,shifts_probe=shifts_probe)



print(np.sum(arr1*np.conj(arr3)))
print(np.sum(arr2*np.conj(arr2)))
print(np.sum(prb1*np.conj(prb3)))


# In[17]:


def line_search(minf, gamma, fu, fu0, fd, fd0):
    """ Line search for the step sizes gamma"""
    while(minf(fu,fu0)-minf(fu+gamma*fd, fu0+gamma*fd0) < 0 and gamma > 0.125/4):
        gamma *= 0.5
    if(gamma <= 0.125/4):  # direction not found
        #print('no direction')
        gamma = 0
    return gamma


# ## $\ \sum_j\sum_i||\mathcal{G}_{d_j}((\mathcal{G}_{d'_j}q)(M_j S_{r_{ij}}\psi_i))|-\sqrt{\text{I}_{ij}}\|^2_2 + \||\mathcal{G}_{d_0}q|-\sqrt{\text{I}_r}\|_2^2\to \text{min}_{\psi_i,q}$ 

# In[18]:


from holotomo.utils import chunk
def adj_holo_batch_ext(pslv,fpsi, data, prb, shifts=None,shifts_probe=None, code=None, shifts_code=None):
    """Batch of Holography transforms"""
    res = np.zeros([ntheta, pslv.ne, pslv.ne], dtype='complex64')
    prb_gpu = cp.array(prb)
    shifts_gpu = None        
    shifts_code_gpu = None
    code_gpu = None

    if code is not None:
        code_gpu = cp.array(code)   
    for ids in chunk(range(pslv.ntheta), pslv.ptheta):
        # copy data part to gpu
        fpsi_gpu = cp.array(fpsi[:, ids])
        data_gpu = cp.array(data[:, ids])
            
        if shifts is not None:
            shifts_gpu = cp.array(shifts[:,ids])
        if shifts_code is not None:
            shifts_code_gpu = cp.array(shifts_code[:,ids])
        if shifts_probe is not None:
            shifts_probe_gpu = cp.array(shifts_probe[:,ids])
        fpsi_gpu = fpsi_gpu-data_gpu*cp.exp(1j*(cp.angle(fpsi_gpu)))        
        # Radon transform
        res_gpu = pslv.adj_holo(fpsi_gpu, prb_gpu, shifts_gpu, code_gpu, shifts_code_gpu,shifts_probe=shifts_probe_gpu)
        # copy result to cpu
        res[ids] = res_gpu.get()
    return res

def adj_holo_prb_batch_ext(pslv, fpsi, data, psi, shifts=None,shifts_probe=None, code=None, shifts_code=None):
        """Batch of Holography transforms"""
        res = np.zeros([len(pslv.distances), pslv.n, pslv.n], dtype='complex64')
        shifts_gpu = None        
        shifts_code_gpu = None
        code_gpu = None
        if code is not None:
            code_gpu = cp.array(code)   
        for ids in chunk(range(pslv.ntheta), pslv.ptheta):
            # copy data part to gpu
            fpsi_gpu = cp.array(fpsi[:, ids])
            psi_gpu = cp.array(psi[ids])
            data_gpu = cp.array(data[:, ids])
            
            if shifts is not None:
                shifts_gpu = cp.array(shifts[:,ids])
            if shifts_code is not None:
                shifts_code_gpu = cp.array(shifts_code[:,ids])
            if shifts_probe is not None:
                shifts_probe_gpu = cp.array(shifts_probe[:,ids])
            # Radon transform
            fpsi_gpu = fpsi_gpu-data_gpu*cp.exp(1j*(cp.angle(fpsi_gpu)))                
            # fprb-data*np.exp(1j*np.angle(fprb))
            res_gpu = pslv.adj_holo_prb(fpsi_gpu, psi_gpu, shifts_gpu,code_gpu,shifts_code_gpu,shifts_probe=shifts_probe_gpu)
            # copy result to cpu
            res += res_gpu.get()
        return res
import time

def cg_holo_batch2(pslv, pslv0, data, data_ref, init, init_prb,  piter,shifts,shifts_probe, shifts_probe_nil, upd_psi=True, upd_prb=False,step=1,vis_step=1,gammapsi0=1,gammaprb0=1):
    """Conjugate gradients method for holography"""

    data = np.sqrt(data)
    data_ref = np.sqrt(data_ref)
    
    # minimization functional
    def minf(fpsi,fprb):
        f = np.linalg.norm(np.abs(fpsi)-data)**2            
        f += np.linalg.norm(np.abs(fprb)-data_ref)**2        
        return f        
    
    psi = init.copy()
    prb = init_prb.copy()
    gammapsi = gammapsi0
    gammaprb = gammaprb0
    
    psi_nil = psi[:1]*0+1
    shifts_nil = shifts[:,:1]*0
    conv = np.zeros([piter])
    tt=np.zeros(10)
    for i in range(piter):
        if upd_psi:
            t = time.time()
            fpsi = pslv.fwd_holo_batch(psi,prb,shifts,shifts_probe=shifts_probe)          
            tt[0] = time.time()-t
            # d = -pslv.adj_holo_batch(fpsi-data*np.exp(1j*(np.angle(fpsi))), prb,shifts)/np.max(np.abs(prb))**2
            t = time.time()
            grad = adj_holo_batch_ext(pslv,fpsi,data, prb,shifts,shifts_probe=shifts_probe)/np.max(np.abs(prb))**2
            tt[1] = time.time()-t
            t = time.time()
            if i == 0 or gammapsi==0:
                d = -grad
            else:
                # d = -grad+np.linalg.norm(grad)**2 / \
                #     ((np.sum(np.conj(d)*(grad-grad0))))*d
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.vdot(d,grad-grad0))*d
            tt[2] = time.time()-t
            grad0 = grad           
            # line search
            t = time.time()
            fd = pslv.fwd_holo_batch(d, prb,shifts,shifts_probe=shifts_probe)                 
            tt[3] = time.time()-t
            t = time.time()
            gammapsi = line_search(minf, gammapsi0, fpsi, 0, fd, 0)
            tt[4] = time.time()-t
            psi = psi+gammapsi*d
            
        if upd_prb:
            t = time.time()
            fprb = pslv.fwd_holo_batch(psi,prb,shifts,shifts_probe=shifts_probe)
            fprb0 = pslv0.fwd_holo_batch(psi_nil,prb,shifts_nil,shifts_probe=shifts_probe_nil)
            tt[5] = time.time()-t
            
            t = time.time()
            gradprb = adj_holo_prb_batch_ext(pslv,fprb,data,psi,shifts,shifts_probe=shifts_probe)#/ndist**2            
            gradprb += adj_holo_prb_batch_ext(pslv0,fprb0,data_ref,psi_nil,shifts_nil,shifts_probe=shifts_probe_nil)
            gradprb *= 1/((pslv.ntheta+1))
            tt[6] = time.time()-t
            t = time.time()
            if i == 0 or gammaprb==0:
                dprb = -gradprb
            else:
                dprb = -gradprb+np.linalg.norm(gradprb)**2 / \
                    (np.vdot(dprb,gradprb-gradprb0))*dprb
            tt[7] = time.time()-t
            gradprb0 = gradprb

            # line search
            t = time.time()
            fdprb = pslv.fwd_holo_batch(psi, dprb,shifts,shifts_probe=shifts_probe)
            fdprb0 = pslv0.fwd_holo_batch(psi_nil, dprb,shifts_nil,shifts_probe=shifts_probe_nil)
            tt[8] = time.time()-t
            t = time.time()
            gammaprb = line_search(minf, gammaprb0, fprb, fprb0, fdprb, fdprb0)
            prb = prb + gammaprb*dprb
            tt[9] = time.time()-t
        if i<3:
            print(np.sum(tt))
            tt=tt/np.sum(tt)*100
            print(f'fwd psi {tt[0]:.2e}, adj psi {tt[1]:.2e}, grad psi {tt[2]:.2e},fwd d {tt[3]:.2e}, line search {tt[4]:.2e}')
            print(f'fwd prb {tt[5]:.2e}, adj prb {tt[6]:.2e}, grad prb {tt[7]:.2e},fwd dprb {tt[8]:.2e}, line search {tt[9]:.2e}')            
            print(np.sum(tt))
        
        if i%vis_step==0:  
            if flg_show:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                im=axs[0].imshow(np.angle(recCTF[0,n//2:-n//2,n//2:-n//2]),cmap='gray')
                axs[0].set_title('reconstructed MultiPaganin phase')
                fig.colorbar(im)
                im=axs[1].imshow(np.angle(psi[0,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2]),cmap='gray')
                axs[1].set_title('reconstructed phase')
                fig.colorbar(im)                
                plt.show()
            dxchange.write_tiff(np.angle(psi),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/r{i:05}.tiff',overwrite=True)
            dxchange.write_tiff(np.angle(psi[0]),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/o{i:05}.tiff',overwrite=True)
            dxchange.write_tiff(np.abs(prb[0]),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/pabs{i:05}.tiff',overwrite=True)
            dxchange.write_tiff(np.angle(prb[0]),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/pang{i:05}.tiff',overwrite=True)        
        
        if i%step==0:
            fprb = pslv.fwd_holo_batch(psi,prb,shifts,shifts_probe=shifts_probe)
            fprb0 = pslv0.fwd_holo_batch(psi_nil,prb,shifts_nil,shifts_probe=shifts_probe_nil)            
            err=minf(fprb,fprb0)
            conv[i]=err
            print(f'{i}) {gammapsi=} {gammaprb=}, {err=:1.5e}', flush=True)
            np.save(f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/conv',conv)
        
            
    return psi,prb,conv


# In[19]:


def cg_holo(pslv, pslv0, data, data_ref, init, init_prb,  piter,shifts,shifts_probe,shifts_probe_nil, upd_psi=True, upd_prb=False,step=1,vis_step=1,gammapsi0=1,gammaprb0=1):
    """Conjugate gradients method for holography"""

    data = cp.sqrt(data)
    data_ref = cp.sqrt(data_ref)

    # minimization functional
    def minf(fpsi,fprb):
        f = cp.linalg.norm(cp.abs(fpsi)-data)**2            
        f += cp.linalg.norm(cp.abs(fprb)-data_ref)**2        
        return f        
    psi = init.copy()
    prb = init_prb.copy()
    gammapsi = gammapsi0
    gammaprb = gammaprb0
    shifts = cp.array(shifts)
    psi_nil = psi[:1]*0+1
    shifts_nil = shifts[:,:1]*0
    conv = np.zeros([piter])
    for i in range(piter):
        if upd_psi:
            
            fpsi = pslv.fwd_holo(psi,prb,shifts,shifts_probe=shifts_probe)          
            grad = pslv.adj_holo(fpsi-data*cp.exp(1j*(cp.angle(fpsi))), prb,shifts,shifts_probe=shifts_probe)/cp.max(cp.abs(prb))**2#/ndist**2
            # d = -grad
            if i == 0 or gammapsi==0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.vdot(d,grad-grad0))*d
            grad0 = grad.copy()           
            fd = pslv.fwd_holo(d, prb,shifts,shifts_probe=shifts_probe)     
            gammapsi = line_search(minf, gammapsi0, fpsi, 0, fd, 0)
      
            psi = psi+gammapsi*d
            
        if upd_prb:
            fprb = pslv.fwd_holo(psi,prb,shifts,shifts_probe=shifts_probe)
            fprb0 = pslv0.fwd_holo(psi_nil,prb,shifts_nil,shifts_probe=shifts_probe_nil)
            gradprb = pslv.adj_holo_prb(fprb-data*cp.exp(1j*cp.angle(fprb)),psi,shifts,shifts_probe=shifts_probe)
            gradprb += pslv0.adj_holo_prb(fprb0-data_ref*cp.exp(1j*cp.angle(fprb0)),psi_nil,shifts_nil,shifts_probe=shifts_probe_nil)
            gradprb*=1/(pslv.ntheta+1)
            # dprb = -gradprb
            if i == 0 or gammaprb==0:
                dprb = -gradprb
            else:
                dprb = -gradprb+cp.linalg.norm(gradprb)**2 / \
                    (cp.vdot(dprb,gradprb-gradprb0))*dprb
            gradprb0=gradprb.copy()
            # line search
            fdprb = pslv.fwd_holo(psi, dprb,shifts,shifts_probe=shifts_probe)
            fdprb0 = pslv0.fwd_holo(psi_nil, dprb,shifts_nil,shifts_probe=shifts_probe_nil)
            
            gammaprb = line_search(minf,gammaprb0, fprb, fprb0, fdprb, fdprb0)
            prb = prb + gammaprb*dprb
            
        if i%step==0:
            fprb = pslv.fwd_holo(psi,prb,shifts,shifts_probe=shifts_probe)
            fprb0 = pslv0.fwd_holo(psi_nil,prb,shifts_nil,shifts_probe=shifts_probe_nil)            
            err=minf(fprb,fprb0)
            conv[i] = err
            print(f'{i}) {gammapsi=} {gammaprb=}, {err=:1.5e}',flush=True)  

        
        if i%vis_step==0:  
            if flg_show:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                im=axs[0].imshow(np.angle(recMultiPaganin[0]),cmap='gray')
                axs[0].set_title('reconstructed recMultiPaganin phase')
                fig.colorbar(im)
                im=axs[1].imshow(cp.angle(psi[0,ne//2-n//2:ne//2+n//2,ne//2-n//2:ne//2+n//2]).get(),cmap='gray')
                # im=axs[1].imshow(cp.angle(psi[0,:,:]).get(),cmap='gray')
                axs[1].set_title('reconstructed phase')
                fig.colorbar(im)                
                plt.show()
                
            dxchange.write_tiff(cp.angle(psi).get(),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{n}_{ntheta}_{ndist}_{st}/r{i:05}.tiff',overwrite=True)
            dxchange.write_tiff(cp.angle(psi[0]).get(),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{n}_{ntheta}_{ndist}_{st}/o{i:05}.tiff',overwrite=True)
            dxchange.write_tiff(np.abs(prb[0]).get(),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/pabs{i:05}.tiff',overwrite=True)
            dxchange.write_tiff(np.angle(prb[0]).get(),f'/data/vnikitin/holo/3d_ald/rfinal_aligned_probe_{same_probe}_{pslv.n}_{ntheta}_{ndist}_{st}/pang{i:05}.tiff',overwrite=True)                                
            
    return psi,prb,conv
def cg_holo_batch(pslv, pslv0, data,data_ref, init, prb_init, piter,shifts=None,shifts_probe=None,shifts_probe_nil=None,upd_psi=True,upd_prb=False,step=1,vis_step=1,gammapsi0=1,gammaprb0=1):
    """Batch of CG solvers"""
    
    res = np.zeros([pslv.ntheta, pslv.ne, pslv.ne], dtype='complex64')
    prb_init_gpu = cp.array(prb_init)                
    shifts_probe_nil_gpu = cp.array(shifts_probe_nil)
    for ids in holotomo.utils.chunk(range(pslv.ntheta), pslv.ptheta):
        # copy data part to gpu
        data_gpu = cp.array(data[:,ids])
        data_ref_gpu = cp.array(data_ref)
        
        init_gpu = cp.array(init[ids])
        shifts_probe_gpu = cp.array(shifts_probe[:,ids])
        # Radon transform
        res_gpu,res_prb_gpu,conv = cg_holo(pslv, pslv0, data_gpu,data_ref_gpu, init_gpu,prb_init_gpu, piter,shifts,
                                           shifts_probe_gpu,shifts_probe_nil_gpu,upd_psi, upd_prb,step,vis_step,gammapsi0,gammaprb0)
        # copy result to cpu
        res[ids] = res_gpu.get()
        res_prb = res_prb_gpu.get()
    return res,res_prb,conv


# In[20]:


pslv = holotomo.SolverHolo(ntheta, n, ne, ptheta, voxelsize, energy, distances, norm_magnifications, distances2,same_probe=same_probe) 
pslv0 = holotomo.SolverHolo(1, n, ne, 1, voxelsize, energy, distances, norm_magnifications, distances2,same_probe=same_probe) 


rec = np.ones([1,ne,ne],dtype='complex64')
rec_prb = np.ones([ndist,n,n],dtype='complex64')        
data_ref = ref0[:,np.newaxis]
shifts_nil = np.array(shifts)[:,:1]*0
_,rec_prb0,_ = cg_holo_batch(pslv0, pslv0, data_ref, data_ref, rec, rec_prb, 17, shifts_nil,shifts_probe,shifts_probe_nil, False,True,1,16,0.5,1)
if flg_show:
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    im=axs[0].imshow(np.abs(rec_prb0[0]),cmap='gray')
    axs[0].set_title('reconstructed amplitude')
    fig.colorbar(im)
    im=axs[1].imshow(np.angle(rec_prb0[0]),cmap='gray')
    axs[1].set_title('reconstructed phase')
    fig.colorbar(im)


# In[21]:


rec = np.ones([ntheta,pslv.ne,pslv.ne],dtype='complex64')
rec = recMultiPaganin.copy()#[:,n-ne//2:n+ne//2,n-ne//2:n+ne//2]#np.pad(recCTF,((0,0),(ne//2-n//2,ne//2-n//2),(ne//2-n//2,ne//2-n//2)),'edge')
data0 = data.copy()
data_ref0 = data_ref.copy()
rec0 = rec.copy()
prb0 = rec_prb0.copy()
# prb0[:] = np.abs(rec_prb0)
n0 = n
ne0 = ne
voxelsize0 = voxelsize
shifts_rec0 = np.array(shifts)
shifts_probe0 = shifts_probe.copy()
shifts_probe_nil0 = shifts_probe_nil.copy()


# In[22]:


import scipy as sp
def downsample(data, binning):
    res = data.copy()
    for k in range(binning):
        res = 0.5*(res[..., ::2,:]+res[..., 1::2,:])
        res = 0.5*(res[..., ::2]+res[..., 1::2])        
    return res

lbinninga = [2,1,0]
iters = [1024,257,257]

# lbinninga = [0]
# iters = [30000]
rec = downsample(rec0,lbinninga[0])
prb = downsample(prb0,lbinninga[0])       
for k,lbinning in enumerate(lbinninga):
    n = n0//2**lbinning
    ne = ne0//2**lbinning
    voxelsize = voxelsize0*2**lbinning
    shifts_rec = shifts_rec0/2**lbinning
    shifts_probe = shifts_probe0/2**lbinning
    shifts_probe_nil = shifts_probe_nil0/2**lbinning
    print(f'{n=},{ne=},{voxelsize=}')   
    
    data = downsample(data0,lbinning)
    data_ref = downsample(data_ref0,lbinning)
    # t0 = cp.array(data_ref[0].astype('complex64'))
    # t0 = cp.tile(t0,(10,1,1))
    # t1 = pslv.apply_shift_complex(t0,cp.array(shifts_probe[1,:10])).real.get()
    # print(np.amax(shifts_probe))
    # plt.figure(figsize=(10,10))
    # plt.imshow(data[1,0]/t1[0],cmap='gray',vmax=1.1,vmin=0.3)    
    # plt.colorbar()
    # plt.show()
    # print(shifts_probe_nil.shape)
    # t0 = cp.array(data_ref[:,0].astype('complex64'))
    # t0 = cp.tile(t0[0],(4,1,1))
    # print(t0.shape)
    # t1 = pslv.apply_shift_complex(t0,0*cp.array(shifts_probe_nil[:,0])).real.get()
    # print(np.amax(shifts_probe))
    # plt.figure(figsize=(10,10))
    # # print(t1.shape)
    # plt.imshow(t1[2]-data_ref[2,0],cmap='gray')#,vmax=1.1,vmin=0.3)    
    # plt.colorbar()
    # plt.show()
    # for k in range(4):
    #     print(np.linalg.norm(t1[k]-data_ref[k,0]))

    # done
    
    pslv = holotomo.SolverHolo(ntheta, n, ne, ptheta, voxelsize, energy, distances, norm_magnifications, distances2,same_probe=same_probe) 
    pslv0 = holotomo.SolverHolo(1, n, ne, 1, voxelsize, energy, distances, norm_magnifications, distances2,same_probe=same_probe)     
    rec,prb,conv = cg_holo_batch2(pslv, pslv0, data, data_ref, rec, prb, iters[k], shifts_rec,shifts_probe,shifts_probe_nil, True,True,iter_step,iter_step, 1,2)
    rec = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(rec)))
    rec = np.pad(rec,((0,0),(rec.shape[1]//2,rec.shape[1]//2),(rec.shape[1]//2,rec.shape[1]//2)))
    rec = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(rec)))*4
    rec = rec.astype('complex64')
    
    prb = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(prb)))
    prb = np.pad(prb,((0,0),(prb.shape[1]//2,prb.shape[1]//2),(prb.shape[1]//2,prb.shape[1]//2)))
    prb = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(prb)))*4
    prb = prb.astype('complex64')
    
    

