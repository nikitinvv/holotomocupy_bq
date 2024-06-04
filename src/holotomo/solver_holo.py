"""Module for tomography."""

import cupy as cp
import numpy as np
from .holo import holo
from .utils import chunk

PLANCK_CONSTANT = 4.135667696e-18  # [keV*s]
SPEED_OF_LIGHT = 299792458  # [m/s]


class SolverHolo():

    def __init__(self, ntheta, n, ne, ptheta, voxelsize, energy, distances, magnification, distances2=None,same_probe=True):
        self.n = n
        self.ne = ne
        self.voxelsize = voxelsize
        self.energy = energy
        self.distances = distances
        self.distances2 = distances2
        self.ntheta = ntheta
        self.ptheta = ptheta
        self.magnification = magnification
        self.same_probe=same_probe

        # Precalculate Fresnel propagators for different distances
        self.fP = cp.zeros([len(distances), 2*n, 2*n], dtype='complex64')
        self.fP2 = cp.zeros([len(distances), 2*n, 2*n], dtype='complex64')
        fx = cp.fft.fftshift(cp.fft.fftfreq(2*n, d=voxelsize))
        [fx, fy] = cp.meshgrid(fx, fx)
        for i, d in enumerate(distances):
            self.fP[i] = cp.exp(-1j*cp.pi*self.wavelength()
                                * d*(fx**2+fy**2))
        if distances2 is not None:
            for i, d in enumerate(distances2):
                self.fP2[i] = cp.exp(-1j*cp.pi *
                                        self.wavelength()*d*(fx**2+fy**2))
        
        # CUDA C class for faster USFFT and padding
        self.cl_holo = holo(ne, ne, n,n, ptheta)
        
    def wavelength(self):
        """Wavelength"""
        return PLANCK_CONSTANT * SPEED_OF_LIGHT / self.energy

    def exptomo(self, psi):
        """Exp representation of projections, exp(i\psi\pi/\lambda)"""
        return np.exp(1j*psi * self.voxelsize * 2*cp.pi / self.wavelength())

    def mlog(self, psi):
        res = psi.copy()
        res[np.abs(psi) < 1e-32] = 1e-32
        res = np.log(res)
        return res

    def logtomo(self, psi):
        """Log representation of projections, -i/\nu log(psi)"""
        return -1j * self.wavelength() / (2*cp.pi) * self.mlog(psi) / self.voxelsize

    def fwd_pad(self, f):
        """Data padding"""

        pad_width = self.n//2
        fpad = cp.zeros([f.shape[0], self.n+2*pad_width,
                        self.n+2*pad_width], dtype='complex64')
        # symmetric padding (CUDA C)
        self.cl_holo.fwd_padsym(fpad.data.ptr, f.data.ptr, pad_width,f.shape[0], 0)
        # fpad = cp.pad(f,((0,0),(pad_width,pad_width),(pad_width,pad_width)),'constant',constant_values=1)
        return fpad/2

    def adj_pad(self, fpad):
        """Adjoint operator for data padding"""

        pad_width = self.n//2
        f = cp.zeros([fpad.shape[0], self.n, self.n], dtype='complex64')
        # Adjoint to symmetric padding
        self.cl_holo.adj_padsym(f.data.ptr, fpad.data.ptr, pad_width,fpad.shape[0], 0)
        # f = fpad[:,pad_width:-pad_width,pad_width:-pad_width]
        return f/2

    def fwd_resample(self, f, magnification):
        # """Data magnification via Fourier domain"""
        # fr = ndimage.zoom(f,(1,magnification/2,magnification/2))
        # st = (fr.shape[1])//2-self.n//2
        # end = st+self.n
        # fr = fr[:,st:end,st:end]
        # fr = fr/self.n/self.n/4
        xk = -((cp.arange(-self.n/2, self.n/2)/self.n) /
               magnification).astype('float32')
        # xk[xk>=0.5] = 0.5-1e-6
        # xk[xk<-0.5] = -0.5
        [xk, yk] = cp.meshgrid(xk, xk)
        fr = cp.zeros([self.ptheta, self.n, self.n], dtype='complex64')
        f = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f,axes=(1,2))),axes=(1,2))
        f = cp.ascontiguousarray(f)
        self.cl_holo.fwd_usfft(fr.data.ptr, f.data.ptr,
                               xk.data.ptr, yk.data.ptr, 0)
        fr = fr/self.ne/self.ne#/4
        return fr

    def adj_resample(self, fr, magnification):
        """Adjoint to data magnification via Fourier domain"""
        # f = cp.zeros([self.ptheta,2*self.n,2*self.n],dtype='complex64')
        # fr = ndimage.zoom(fr,(1,2/magnification,2/magnification))

        # st = self.n-fr.shape[1]//2
        # end = self.n-fr.shape[1]//2+fr.shape[1]

        # f[:,st:end,st:end]=fr*(magnification/2)**2
        xk = -((cp.arange(-self.n/2, self.n/2)/self.n) /
               magnification).astype('float32')
        # xk[xk>=0.5] = 0.5-1e-6
        # xk[xk<-0.5] = -0.5

        [xk, yk] = cp.meshgrid(xk, xk)
        f = cp.zeros([self.ptheta, self.ne, self.ne], dtype='complex64')
        fr = cp.ascontiguousarray(fr)
        self.cl_holo.adj_usfft(f.data.ptr, fr.data.ptr,
                               xk.data.ptr, yk.data.ptr, 0)
        f = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f,axes=(1,2))),axes=(1,2))
        return f

    def fwd_propagate(self, f, fP):
        """Fresnel transform"""

        ff = self.fwd_pad(f)
        # ff=f.copy()
        ff = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(ff,axes=(1,2))),axes=(1,2))
        ff = ff*fP
        ff = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(ff,axes=(1,2))),axes=(1,2))
        # if self.ne==self.n*2:
        ff = self.adj_pad(ff)
        return ff

    def adj_propagate(self, ff, fP):
        """Adjoint to Fresnel transform"""
        # if self.ne==self.n*2:
        f = self.fwd_pad(ff)
        # else:
        # f=ff.copy()
        f = cp.fft.fftshift(cp.fft.fft2(cp.fft.fftshift(f,axes=(1,2))),axes=(1,2))
        f = f*cp.conj(fP)
        f = cp.fft.fftshift(cp.fft.ifft2(cp.fft.fftshift(f,axes=(1,2))),axes=(1,2))
        # if self.ne==self.n*2:
        f = self.adj_pad(f)
        return f

    def apply_shift_complex(self, psi, p):
        """Apply shift for all projections."""
        n0 = psi.shape[-1]
        # if n0!=2*self.n:
        #     psi = cp.pad(psi,((0,0),(self.n//2,self.n//2),(self.n//2,self.n//2)),'symmetric')
        if len(p.shape) == 1:
            p = p[np.newaxis]
        [x, y] = cp.meshgrid(cp.fft.fftfreq(n0),
                             cp.fft.fftfreq(n0))
        shift = cp.exp(-2*cp.pi*1j *
                       (x*p[:, 1, None, None]+y*p[:, 0, None, None])).astype('complex64')
        res = cp.fft.ifft2(shift*np.fft.fft2(psi))
        return res

    def fwd_holo(self, psi, prb, shift=None, code=None, shift_code=None,shifts_probe=None):
        """holography transform: padding, magnification, multiplication by probe, Fresnel transform"""
        data = cp.zeros([len(self.distances), self.ptheta,
                        self.n, self.n], dtype='complex64')
        
        for i in range(len(self.distances)):
            prbr = prb[i:i+1].copy()
            psir = psi.copy()
            if shift_code is not None:    # shift in scaled coordinates
                coder = self.apply_shift_complex(code, shift_code[i])
                # print(coder.shape)
                coder = coder[:,self.n//2:3*self.n//2,self.n//2:3*self.n//2]
            if code is not None: # multiple the code and probe                
                prbr = prbr*coder            
            if self.distances2 is not None:  # propagate the probe from plane 0 to plane i
                if shifts_probe is not None:
                    prbr = cp.tile(prbr,(self.ptheta,1,1))
                    prbr = self.apply_shift_complex(prbr, shifts_probe[i])
                prbr = self.fwd_propagate(prbr, self.fP2[i])
            if shift is not None:    # shift in scaled coordinates
                psir = self.apply_shift_complex(psir, shift[i])
            # # scale object
            if self.ne!=self.n:
                psir = self.fwd_resample(psir, self.magnification[i]*self.ne/self.n)                
            # # multiply the probe and object
            psir *= prbr            
            # propagate both
            psir = self.fwd_propagate(psir, self.fP[i])
            
            data[i] = psir
        # data /= len(self.distances)
        return data

    def adj_holo(self, data, prb, shift=None, code=None, shift_code=None, shifts_probe=None):
        """Adjoint holography transform wrt object (adjoint operations in reverse order))"""

        psi = cp.zeros([self.ptheta, self.ne, self.ne], dtype='complex64')
        
        for i in range(len(self.distances)):
            psir = data[i].copy()
            prbr = prb[i:i+1].copy()
            if shift_code is not None:    # shift code
                coder = self.apply_shift_complex(code, shift_code[i])
                coder = coder[:,self.n//2:3*self.n//2,self.n//2:3*self.n//2]
                        
            if code is not None:
                prbr = prbr*coder                        
            # propagate data back
            psir = self.adj_propagate(psir, self.fP[i])
            
            if self.distances2 is not None:  # propagate the probe from plane 0 to plane i
                if shifts_probe is not None:
                    prbr = cp.tile(prbr,(self.ptheta,1,1))
                    prbr = self.apply_shift_complex(prbr, shifts_probe[i])
                prbr = self.fwd_propagate(prbr, self.fP2[i])
            # multiply the conj probe and object
            psir *= cp.conj(prbr)
            
            # scale object
            if self.ne!=self.n:
                psir = self.adj_resample(psir, self.magnification[i]*self.ne/self.n)
                
            # psir = cp.pad(psir,((0,0),(self.n//2,self.n//2),(self.n//2,self.n//2)))
            if shift is not None:  # shift object back
                psir = self.apply_shift_complex(psir, -shift[i])
            
            psi += psir
        # psi /= len(self.distances)
        return psi

    def adj_holo_prb(self, data, psi,  shift=None, code=None, shift_code=None,shifts_probe=None):
        """Adjoint holography transform wrt object (adjoint operations in reverse order))"""
        prb = cp.zeros([len(self.distances), self.n, self.n], dtype='complex64')
        
        for i in range(len(self.distances)):
            prbr = data[i].copy()
            psir = psi.copy()
            # coder = code.copy()
            # propagate data back
            prbr = self.adj_propagate(prbr, self.fP[i])
            if shift is not None:
                # shift in scaled coordinates
                psir = self.apply_shift_complex(psir, shift[i])
            # scale object
            if self.ne!=self.n:
                psir = self.fwd_resample(psir, self.magnification[i]*self.ne/self.n)
            # multiply the probe and conj scaled object
            # probably sum over angles needed
            prbr *= cp.conj(psir)
            if self.distances2 is not None:
                prbr = self.adj_propagate(prbr, self.fP2[i])
                if shifts_probe is not None:
                    prbr = self.apply_shift_complex(prbr, -shifts_probe[i])
            
            if shift_code is not None:    # shift code
                coder = self.apply_shift_complex(code, shift_code[i])                        
                coder = coder[:,self.n//2:3*self.n//2,self.n//2:3*self.n//2]
            if code is not None:
                prbr = prbr*cp.conj(coder)                                       
           # prb += cp.sum(prbr,axis=0)
            prb[i:i+1] = cp.sum(prbr,axis=0)# to be consistent with what follows
        #print(self.same_probe)
        if self.same_probe:
            prb[:] = np.sum(prb,axis=0)/len(self.distances)
        return prb

    def fwd_holo_batch(self, psi, prb,  shifts=None, code=None, shifts_code=None,shifts_probe=None):
        """Batch of Holography transforms"""
        res = np.zeros([len(self.distances), self.ntheta,
                       self.n, self.n], dtype='complex64')
        prb_gpu = cp.array(prb)
        
        shifts_gpu = None        
        shifts_code_gpu = None
        code_gpu = None
        shifts_probe_gpu = None

        if code is not None:
            code_gpu = cp.array(code)        
        
        for ids in chunk(range(self.ntheta), self.ptheta):
            # copy data part to gpu
            psi_gpu = cp.array(psi[ids])
            if shifts is not None:
                shifts_gpu = cp.array(shifts[:,ids])
            if shifts_code is not None:
                shifts_code_gpu = cp.array(shifts_code[:,ids])
            if shifts_probe is not None:
                shifts_probe_gpu=cp.array(shifts_probe[:,ids])    
            # Radon transform
            res_gpu = self.fwd_holo(psi_gpu, prb_gpu, shifts_gpu, code_gpu, shifts_code_gpu,shifts_probe_gpu)
            # copy result to cpu
            res[:, ids] = res_gpu.get()
        return res

    def adj_holo_batch(self, fpsi, prb, shifts=None, code=None, shifts_code=None,shifts_probe=None):
        """Batch of Holography transforms"""
        res = np.zeros([self.ntheta, self.ne, self.ne], dtype='complex64')
        prb_gpu = cp.array(prb)
        shifts_gpu = None        
        shifts_code_gpu = None
        code_gpu = None
        shifts_probe_gpu = None

        if code is not None:
            code_gpu = cp.array(code)   
        for ids in chunk(range(self.ntheta), self.ptheta):
            # copy data part to gpu
            fpsi_gpu = cp.array(fpsi[:, ids])
            if shifts is not None:
                shifts_gpu = cp.array(shifts[:,ids])
            if shifts_code is not None:
                shifts_code_gpu = cp.array(shifts_code[:,ids])
            if shifts_probe is not None:
                shifts_probe_gpu=cp.array(shifts_probe[:,ids])  
            # Radon transform
            res_gpu = self.adj_holo(fpsi_gpu, prb_gpu, shifts_gpu, code_gpu, shifts_code_gpu,shifts_probe_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def adj_holo_prb_batch(self, fpsi, psi, shifts=None, code=None, shifts_code=None,shifts_probe=None):
        """Batch of Holography transforms"""
        res = np.zeros([len(self.distances), self.n, self.n], dtype='complex64')
        shifts_gpu = None        
        shifts_code_gpu = None
        code_gpu = None
        shifts_probe_gpu = None
        
        if code is not None:
            code_gpu = cp.array(code)   
        for ids in chunk(range(self.ntheta), self.ptheta):
            # copy data part to gpu
            fpsi_gpu = cp.array(fpsi[:, ids])
            psi_gpu = cp.array(psi[ids])
            
            if shifts is not None:
                shifts_gpu = cp.array(shifts[:,ids])
            if shifts_code is not None:
                shifts_code_gpu = cp.array(shifts_code[:,ids])
            if shifts_probe is not None:
                shifts_probe_gpu=cp.array(shifts_probe[:,ids])  
            
            # Radon transform
            res_gpu = self.adj_holo_prb(fpsi_gpu, psi_gpu, shifts_gpu,code_gpu,shifts_code_gpu,shifts_probe_gpu)
            # copy result to cpu
            res += res_gpu.get()
        return res
