__author__ = 'neural22'

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

#Plot data result with matplot lib
def plot_result(data, output=None):
    plt.subplot(2, 1, 2)
    plt.plot(data, 'b-', label='data')
    if not output is None:
        plt.plot(output, 'g-', linewidth=2, label='filtered data')
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.35)
    plt.show()
    
# Abstract class to create digital filters with a common interface
# All parameters should be passed via kwargs
class DigitalFilter:

    def __init__(self, **kwargs):
        self.order = kwargs.get('order', 1)
        self.start = kwargs.get('start', 0)
        self.end = kwargs.get('end', 0)
        self.filter_type = kwargs.get('filter_type', None)
        self.fs = kwargs.get('fs', 1)
        self.rp = kwargs.get('rp', 1)
        self.rs = kwargs.get('rs', 1)
        self.kernel_size = kwargs.get('kernel_size', 0)
        self.no_nyq = kwargs.get('no_nyq', False)
        self.description = 'Digital Filter'
        self.num, self.den = self.create_filter()

    # it return filter if form of (numerator, denominator)
    def create_filter(self):
        return 0, 1

    # Apply filter to data, return the list with the new values
    def apply(self, data):
        return signal.lfilter(self.num, self.den, data)
    
    # Apply filter to data with forward-backward filter for linear phase, return the list with the new values
    def apply_forward_backward(self, data):
        return signal.filtfilt(self.num, self.den, data)

    def compute_freq_response(self):
        return signal.freqz(self.num, self.den)

    # Plot the response in frequency
    def plot(self, print_phase=False):
        w, h = self.compute_freq_response()
        fig = plt.figure()
        plt.title('Digital filter frequency response')
        ax1 = fig.add_subplot(111)
        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        # plot lines -40dB
        h_ref = np.array([100 for i in h])
        plt.plot(w, -20 * np.log10(abs(h_ref)), 'r')
        if print_phase:
            ax2 = ax1.twinx()
            angles = np.unwrap(np.angle(h))
            plt.plot(w, angles, 'g')
            plt.ylabel('Angle (radians)', color='g')
            plt.grid()
            plt.axis('tight')
        plt.show()

# Test object
class TestFilter:

    def __init__(self, time):
        self.T = time
        self.n = 0
        self.t = None
        self.data = None
        self.output = 0

    # Test the filter f with the function sin(pi * e^t)
    # chk_bf: if True apply_forward_backward is used, for linear phase
    def test_filter(self, f, chk_bf=False):
        # total number of samples
        self.n = int(self.T * f.fs)
        self.t = np.linspace(0, self.T, self.n, endpoint=False)
        self.data = np.sin(np.pi * np.e**(self.t))
        if chk_bf:
            self.output = f.apply_forward_backward(self.data)
        else:
            self.output = f.apply(self.data)
        plot_result(self.data, self.output)

    # Test the filter f applying data, plot the result in the same graphic
    def test_filter_with_data(self, f, data, chk_bf=False):
        if chk_bf:
            output = f.apply_forward_backward(data)
        else:
            output = f.apply(data)
        plot_result(data, output)
    
    # Test the filter f applying data, plot the result in the same graphic + plot an other graphics with only the result
    # (data scaled)
    def test_filter_with_data_splitted(self, f, data, chk_bf=False):
        self.test_filter_with_data(f, data)
        if chk_bf:
            output = f.apply_forward_backward(data)
        else:
            output = f.apply(data)
        plot_result(output)
        
## Common IIR Filters

class EllipticLow(DigitalFilter):

    def create_filter(self):
        self.description = 'Elliptinc Low Pass Filter:' + str(self.start)
        self.filter_type = 'lowpass'
        #Nyquist frequency
        if not self.no_nyq:
            nyq = 0.5 * self.fs
            normal_cutoff = self.start / nyq
        else:
            normal_cutoff = self.start
        return signal.ellip(self.order, self.rp, self.rs, normal_cutoff, self.filter_type)

    
class EllipticHigh(DigitalFilter):

    def create_filter(self):
        self.description = 'Elliptinc High Pass Filter:' +  str(self.end)
        self.filter_type = 'highpass'
        if not self.no_nyq:
            #Nyquist frequency
            nyq = 0.5 * self.fs
            normal_cutoff = self.end / nyq
        else:
            normal_cutoff = self.end
        return signal.ellip(self.order, self.rp, self.rs, normal_cutoff, self.filter_type)


class EllipticBandPass(DigitalFilter):

    def create_filter(self):
        self.description = 'Elliptic Band Pass Filter:' + str(self.start) + '-' + str(self.end)v
        self.filter_type = 'bandpass'
        if if not self.no_nyq:
            nyq = 0.5 * self.fs
            normal_cutoff_start = self.start / nyq
            normal_cutoff_end = self.end / nyq
        else:
            normal_cutoff_start = self.start
            normal_cutoff_end = self.end 
        return signal.ellip(self.order, self.rp, self.rs, (normal_cutoff_start, normal_cutoff_end), self.filter_type)
    
    
class MedianFilter(DigitalFilter):

    def apply(self, data):
        if self.kernel_size:
            return signal.medfilt(data, kernel_size=self.kernel_size)
        else:
            return signal.medfilt(data)
        
        
class ButterworthLow(DigitalFilter):

    def create_filter(self):
        self.description = 'Butterworth Low Pass Filter:' + str(self.start)
        self.filter_type = 'lowpass'
        if not self.no_nyq:
            #Nyquist frequency
            nyq = 0.5 * self.fs
            normal_cutoff = self.start / nyq
        else:
            normal_cutoff = self.start
        return signal.butter(self.order, normal_cutoff, self.filter_type)
    

class ButterworthHigh(DigitalFilter):

    def create_filter(self):
        self.description = 'Butterworth High Pass Filter:' + str(self.end)
        self.filter_type = 'highpass'
        if not self.no_nyq:
            nyq = 0.5 * self.fs
            normal_cutoff = self.end / nyq
        else:
            normal_cutoff = self.end
        return signal.butter(self.order, normal_cutoff, self.filter_type)
    
class ButterworthBandPass(DigitalFilter):

    def create_filter(self):
        self.description = 'Butterworth Band Pass Filter: ' + str(self.start) + '-' + str(self.end)
        self.filter_type = 'bandpass'
        if not self.no_nyq:
            nyq = 0.5 * self.fs
            normal_cutoff_start = self.start / nyq
            normal_cutoff_end = self.end / nyq
        else:
            normal_cutoff_start = self.start
            normal_cutoff_end = self.end
        return signal.butter(self.order, (normal_cutoff_start, normal_cutoff_end), self.filter_type)

class ChebyshevBandPass(DigitalFilter):

    def create_filter(self):
        self.description = 'Chebyshev Band Pass Filter: ' + str(self.start) + '-' + str(self.end)
        self.filter_type = 'bandpass'
        if not self.no_nyq:
            nyq = 0.5 * self.fs
            normal_cutoff_start = self.start / nyq
            normal_cutoff_end = self.end / nyq
        else:
            normal_cutoff_start = self.start
            normal_cutoff_end = self.end
        return signal.cheby1(self.order, self.rp, (normal_cutoff_start, normal_cutoff_end), self.filter_type)

# FIR Filters
class FIRLowPassFilter(DigitalFilter):
    def create_filter(self):
        self.description = 'FIR Low Pass Filter: ' + str(self.start)
        self.filter_type = 'bandpass'
        nyq = 0.5 * self.fs
        normal_cutoff_start = self.start / nyq
        return signal.firwin(self.order+1, normal_cutoff_start), [1.0]    

class FIRHighPassFilter(DigitalFilter):
    def create_filter(self):
        self.description = 'FIR High Pass Filter: ' + str(self.end)
        self.filter_type = 'bandpass'
        nyq = 0.5 * self.fs
        normal_cutoff_end = self.end / nyq
        return signal.firwin(self.order+1, normal_cutoff_end, pass_zero=False), [1.0]
    
class FIRBandPassFilter(DigitalFilter):
    def create_filter(self):
        self.description = 'FIR Band Pass Filter: ' + str(self.start) + '-' + str(self.end)
        self.filter_type = 'bandpass'
        nyq = 0.5 * self.fs
        normal_cutoff_start = self.start / nyq
        normal_cutoff_end = self.end / nyq
        return signal.firwin(self.order+1, [normal_cutoff_start, normal_cutoff_end], pass_zero=False), [1.0]
