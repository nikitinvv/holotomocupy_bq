#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import dxchange
import cupy as cp
import numpy as np
import tomoalign


# # Init data sizes and parametes of the PXM of ID16A

# In[ ]:


ntheta = 1500
n = 1536
ne = 2528
nz = 1280
ndist = 4
ngpus = 4
pnz = 64
same_prb = True
center = 791
shiftc = int(center-n//2)


# In[ ]:


def apply_shift(psi, p):
    """Apply shift for all projections."""
    psi = cp.array(psi)
    p = cp.array(p)
    [nz,n] = psi.shape[1:]
    tmp = cp.pad(psi,((0,0),(nz//2,nz//2),(n//2,n//2)), 'symmetric')
    [x, y] = cp.meshgrid(cp.fft.rfftfreq(2*n),
                         cp.fft.fftfreq(2*nz))
    shift = cp.exp(-2*cp.pi*1j *
                   (x*p[:, 1, None, None]+y*p[:, 0, None, None]))
    res0 = cp.fft.irfft2(shift*cp.fft.rfft2(tmp))
    res = res0[:, nz//2:3*nz//2, n//2:3*n//2].get()
    return res

shifts = np.zeros([1500,2])
shifts[:,0] = -np.linspace(0,5,1500)
shifts[:,1] = 0 

iter = 192
data = np.zeros([1500,nz,n],dtype='float32')


theta = np.loadtxt(f'/data/viktor/id16a/3d_ald4/3d_ald4_ht_10nm_/angles_file.txt').astype('float32')[:]/180*np.pi

for st in range(0,1500//250):
    data[st:1500:1500//250] = dxchange.read_tiff(f'/data/vnikitin/holo/3d_ald/rfinal_probe_{same_prb}_{n}_{250}_{ndist}_{st}/r{iter:05}.tiff')[:,800:800+nz,ne//2-n//2+shiftc:ne//2+n//2+shiftc]
    
data_new = data.copy()
for k in range(ntheta):
    data_new[k:k+1] = apply_shift(data[k:k+1],-shifts[k:k+1])
dxchange.write_tiff_stack(data_new,f'/data/vnikitin/holo/3d_ald/data_new/3.tiff',overwrite=True)

init = np.zeros([nz,n,n],dtype='float32')
with tomoalign.SolverTomo(theta, ntheta, nz, n, pnz, n/2, ngpus) as tslv:
    u = tslv.cg_tomo_batch(data_new, init, 400)
    dxchange.write_tiff(u, f'/data/vnikitin/holo/3d_ald/rfinal_probe_{same_prb}_{n}_{ntheta}_{ndist}/cgfinal/r_{400:04}',overwrite=True)


# noprobe
iter = 144
for st in range(0,1500//250):
    data[st:1500:1500//250] = dxchange.read_tiff(f'/data/vnikitin/holo/3d_ald/rfinal_probe_{same_prb}_{n}_{250}_{ndist}_{st}/r{iter:05}noprobe.tiff')[:,800:800+nz,ne//2-n//2+shiftc:ne//2+n//2+shiftc]
    
data_new = data.copy()
for k in range(ntheta):
    data_new[k:k+1] = apply_shift(data[k:k+1],-shifts[k:k+1])
dxchange.write_tiff_stack(data_new,f'/data/vnikitin/holo/3d_ald/data_new/3noprobe.tiff',overwrite=True)

init = np.zeros([nz,n,n],dtype='float32')
with tomoalign.SolverTomo(theta, ntheta, nz, n, pnz, n/2, ngpus) as tslv:
    u = tslv.cg_tomo_batch(data_new, init, 400)
    dxchange.write_tiff(u, f'/data/vnikitin/holo/3d_ald/rfinal_probe_{same_prb}_{n}_{ntheta}_{ndist}/cgfinalnoprobe/r_{400:04}',overwrite=True)
