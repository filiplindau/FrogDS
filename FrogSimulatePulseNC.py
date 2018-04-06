'''
Created on 3 Feb 2016

@author: Filip Lindau

Calculate FROG with carrier frequency removed
'''

import numpy as np
import logging
from scipy.interpolate import interp1d


root = logging.getLogger()
while len(root.handlers):
    root.removeHandler(root.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
root.addHandler(fh)
root.setLevel(logging.DEBUG)


class SimulatedPulse(object):
    def __init__(self, N=128, dt=10e-15, l0=800e-9, tau=50e-15):
        """

        :param N: number of points in the trace
        :param dt: time resolution
        :param l0: central wavelength of the peak
        :param tau: tentative pulse duration
        """
        self.Et = np.zeros(N, dtype=np.complex)
        self.N = N
        self.tau = tau
        self.l0 = l0
        self.tspan = N*dt
        self.dt = dt

        # Now we calculate the frequency resolution required by the
        # time span
        self.w_res = 2 * np.pi / self.tspan

        # Frequency span is given by the time resolution
        f_max = 1 / (2 * self.dt)
        self.w_span = f_max * 2 * 2 * np.pi

        c = 299792458.0
        w0 = 2 * np.pi * c / self.l0
        # delta_l = -2*np.pi*c/self.w_span + np.sqrt(l0**2 + (2*np.pi*c/self.w_span)**2)
        # w0 = 2*np.pi*c*l0/(l0**2 - delta_l**2)
        self.w0 = w0

        self.generateGaussian(tau)
        
    def generateGaussian(self, tau):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + ph) * np.exp(0j)
        self.setEt(Eenv, t)
        
    def generateGaussianQuadraticPhase(self, b=0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*(b*t**2) + ph)
        self.setEt(Eenv, t)

    def generateGaussianCubicPhase(self, b=0, c=0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-t**2/self.tau**2 + 1j*(b*t**2 + c*t**3) + ph)
        self.setEt(Eenv, t)

    def generateGaussianCubicSpectralPhase(self, b=0, c=0):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)        
        f_max = 1/(2*self.dt)
        w_span = f_max*2*2*np.pi
        w = np.linspace(-w_span/2, w_span/2, self.N)
        dw = 2*np.pi*0.441/self.tau
        ph = 0.0
        Eenv_w = np.exp(-w**2/dw**2 + 1j*(b*w**2 + c*w**3) + ph)
        Eenv = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Eenv_w)))
        self.setEt(Eenv, t)
        
    def generateDoublePulse(self, tau, deltaT):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)
        ph = 0.0
        Eenv = np.exp(-(t+deltaT/2)**2/self.tau**2 + ph) + np.exp(-(t-deltaT/2)**2/self.tau**2 + ph) 
        self.setEt(Eenv, t)
        
    def addChirp(self, a):
        t = np.linspace(-self.tspan/2, self.tspan/2, self.N)                
        Eenv = self.Et.copy()*np.exp(1j*a*t*t)
        self.setEt(Eenv, t)        
        
    def getFreqSpectrum(self, Nw=None):
        if Nw is None:
            Nw = self.t.shape[0]
        Ew = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et), Nw))
#        w0 = 0*2*np.pi*299792458.0/self.l0
        w = np.fft.fftshift((self.w0 + 2*np.pi*np.fft.fftfreq(Nw, d=self.dt)))
        return w, Ew

    def getSpectrogram(self, Nl=None):
        if Nl is None:
            Nl = self.t.shape[0]
        w, Ew = self.getFreqSpectrum(Nl)
        c = 299792458.0
        l = 2*np.pi*c/w

        # Resample over linear grid in wavelength
        E_interp = interp1d(l, Ew, kind='linear', fill_value=0.0, bounds_error=False)
        l_start = 2*np.pi*c / (self.w0 + self.w_span/2)
        l_stop = 2 * np.pi * c / (self.w0 - self.w_span/2)
        delta_l = np.min([np.abs(self.l0-l_start), np.abs(self.l0-l_stop)])
        l_sample = np.linspace(self.l0-delta_l, self.l0+delta_l, Nl)
        El = E_interp(l_sample)
        Il = El*El.conj()
#        Il = Il/max(Il)
        return l_sample, Il

    def setEt(self, Et, t=None):
        """
        Manually electric field vector. If no time vector is supplied, the internal is used. Then the
        new vector must have the same number of elements.
        :param Et: New electric field vector
        :param t: New time vector or None
        :return:
        """
        if t is not None:
            self.t = t
            self.tspan = np.abs(t.max()-t.min())
            self.N = t.shape[0]
            self.dt = self.tspan/self.N
        self.Et = Et
        
    def getShiftedEt(self, shift):
        """
        Return shifted electric field vector. Useful for generating time delays
        :param shift: Number of timesteps to shift the vector
        :return: Shifted electroc field. Exposed edges are filled with zeros.
        """
        sh = np.int(shift)
        Ets = np.zeros_like(self.Et)
        # root.debug(''.join(('Shift: ', str(sh))))
        if sh < 0:
            Ets[0:self.N+sh] = self.Et[-sh:]
        else:
            Ets[sh:] = self.Et[0:self.N-sh]
        return Ets


class SimulatedFrogTrace(object):
    def __init__(self, N=128, dt=10e-15, l0=800e-9, tau=50e-15):
        self.pulse = SimulatedPulse(N, dt, l0, tau)
        self.l_vec = None
        self.tau_vec = None
        
    def generateSHGTraceDt(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0/2, self.pulse.tau)
        tspan = N*dt
        self.tau_vec = np.linspace(-tspan/2.0, tspan/2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []
        
        signalPulse.setEt(Et*Et, t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift/np.abs(l[1]-l[0]))
        self.l_vec = l + l_shift
        
        shiftVec = np.arange(N) - N/2
        
        root.debug(''.join(('l_shift: ', str(l_shift))))
        root.debug(''.join(('nl_shift: ', str(nl_shift))))
        
        for sh in shiftVec:
            Ils = np.zeros_like(Il)
            signalPulse.setEt(Et*self.pulse.getShiftedEt(sh), t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            Ils = np.roll(Iln, nl_shift)
            Ifrog.append(Ils)
                        
        Ifrog = np.array(Ifrog).real
        self.Ifrog = Ifrog/Ifrog.max()
        
        return self.Ifrog

    def generateSDTraceDt(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0, self.pulse.tau)
        tspan = N*dt
        self.tau_vec = np.linspace(-tspan/2.0, tspan/2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []
        
        signalPulse.setEt(Et*Et*np.conj(Et), t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift/np.abs(l[1]-l[0]))
        self.l_vec = l + l_shift
        
        shiftVec = (np.arange(N) - N/2).astype(np.int)
        
        root.debug(''.join(('l_shift: ', str(l_shift))))
        root.debug(''.join(('nl_shift: ', str(nl_shift))))
        
        Esig_t_tau = []
        for sh in shiftVec:
            Ils = np.zeros_like(Il)
            signalPulse.setEt(Et*Et*np.conj(self.pulse.getShiftedEt(sh)), t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            Ils = np.roll(Iln, nl_shift)
            Ifrog.append(Ils)
            Esig_t_tau.append(signalPulse.Et)
            
        Ifrog = np.array(Ifrog).real
        self.Ifrog = Ifrog/Ifrog.max()       
        self.Esig_t_tau = np.array(Esig_t_tau) 
        
        return self.Ifrog
    
    def generatePGTraceDt(self, N, dt, l0):
        signalPulse = SimulatedPulse(self.pulse.N, self.pulse.dt, self.pulse.l0, self.pulse.tau)
        tspan = N * dt
        self.tau_vec = np.linspace(-tspan / 2.0, tspan / 2.0, N)
        t = self.pulse.t
        Et = self.pulse.Et
        Ifrog = []

        signalPulse.setEt(Et * Et * np.conj(Et), t)
        l, Il = signalPulse.getSpectrogram(t.shape[0])
        l_shift = signalPulse.l0 - l0
        nl_shift = np.int(l_shift / np.abs(l[1] - l[0]))
        self.l_vec = l + l_shift

        shiftVec = (np.arange(N) - N / 2).astype(np.int)

        root.debug(''.join(('l_shift: ', str(l_shift))))
        root.debug(''.join(('nl_shift: ', str(nl_shift))))

        Esig_t_tau = []
        for sh in shiftVec:
            Ils = np.zeros_like(Il)
            signalPulse.setEt(Et*np.abs(self.pulse.getShiftedEt(sh))**2, t)
            l, Iln = signalPulse.getSpectrogram(t.shape[0])
            Ils = np.roll(Iln, nl_shift)
            Ifrog.append(Ils)
            Esig_t_tau.append(signalPulse.Et)

        Ifrog = np.array(Ifrog).real
        self.Ifrog = Ifrog / Ifrog.max()
        self.Esig_t_tau = np.array(Esig_t_tau)

        return self.Ifrog
        
    def addNoise(self, shotAmp=0.1, readAmp=0.05):
        self.Ifrog = np.maximum(0, self.Ifrog + 
                                np.random.poisson(self.Ifrog/shotAmp)*shotAmp + 
                                np.random.standard_normal(self.Ifrog.shape)*readAmp)
        return self.Ifrog
    
    def getWavelengths(self):
        return self.l_vec
    
    def getTimedelays(self):
        return self.tau_vec
    
if __name__ == '__main__':
    Ns = 512
    dt = 5e-15
    tau = 100e-15 / (2*np.sqrt(np.log(2)))
    l0 = 263.5e-9
    p = SimulatedPulse(Ns, dt, l0, tau)
#     p.generateGaussianCubicPhase(5e24, 1e40)
#     p.generateGaussianCubicPhase(-5e26, 0)
    p.generateGaussianCubicSpectralPhase(0e-27, 0.3e-40)
#     p.generateDoublePulse(tau, 0.5e-12)
#     p.generateGaussian(tau)
#     p.addChirp(1e26)
    gt = SimulatedFrogTrace(Ns, dt, l0)
    gt.pulse = p
    IfrogSHG = gt.generateSHGTraceDt(Ns, dt, l0/2)
    IfrogSD = gt.generateSDTraceDt(Ns, dt, l0)
    IfrogPG = gt.generatePGTraceDt(Ns, dt, l0)
    l = gt.getWavelengths()
    t = gt.getTimedelays()
