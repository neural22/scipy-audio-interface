__author__ = 'aloriga'

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
#import pickle
from utils import SerializableObject

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
class DigitalFilter(SerializableObject):

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

    def create_filter(self):
        """
        Create filter
        :return: tuple (numerator, denominator)
        """
        return 0, 1

    def apply(self, data):
        """
        Apply filter to data
        :param data: array to apply the filter
        :return: array with new values
        """
        return signal.lfilter(self.num, self.den, data)

    def apply_forward_backward(self, data):
        """
        Apply filter to data with forward-backward filter
        :param data: array to apply the filter
        :return: array with new values
        """
        return signal.filtfilt(self.num, self.den, data)

    def compute_freq_response(self):
        return signal.freqz(self.num, self.den)

    # Plot the response in frequency
    def plot(self, print_phase=False):
        """
        Plot the frequency response of the DigitalFilter
        :param print_phase: boolean, if true phase response is plotted
        :return:
        """
        w, h = self.compute_freq_response()
        fig = plt.figure()
        plt.title('Digital filter frequency response')
        ax1 = fig.add_subplot(111)
        plt.plot(w, 20 * np.log10(abs(h)), 'b')
        plt.ylim([-150, 10])
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


# FILTERS DESIGN
# Common IIR Filters
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
        self.description = 'Elliptic Band Pass Filter:' + str(self.start) + '-' + str(self.end)
        self.filter_type = 'bandpass'
        if not self.no_nyq:
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


# MULTIRATE FILTER BANK
class FilterBank(SerializableObject):

    def __init__(self, filters, sampling_fs=44100):
        """
        :param filters: dictionary, key: filter name, value: DigitalFilter
        :param sampling_fs: sampling frequency
        :return:
        """
        # check if filters are DigitalFilter objects
        assert reduce(lambda x, y: x and y, [isinstance(filters[f], DigitalFilter) for f in filters])
        self.filters = filters
        self.sampling_fs = sampling_fs
       
    def apply(self, initial_data, resample=False):
        """
        Apply all filters to data
        :param initial_data: array to apply the filter
        :param resample: boolean, if True resample method is applied. See multi-rate filtering
        :return: data processed
        """
        return self._apply_function(initial_data, resample, 'apply')
       
    def apply_forward_backward(self, initial_data, resample=False):
        """
        Apply all filters to data with backward-forward method
        :param initial_data: array to apply the filter
        :param resample: boolean, if True resample method is applied. See multi-rate filtering
        :return: data processed
        """
        return self._apply_function(initial_data, resample, 'apply_forward_backward') 
        
    def _apply_function(self, initial_data, resample, function):
        output = {}
        output_resampled = {}
        nan_errors = []
        for f in self.filters:
            decimate_factor = self.sampling_fs/self.filters[f].fs
            data = signal.decimate(initial_data, decimate_factor)
            # apply function
            output[f] = getattr(self.filters[f], function)(data)
            # normalize output
            output[f] = output[f]/max(output[f])
            if resample:
                # resample with FFT, it can be slow!
                output_resampled[f] = signal.resample(output[f], len(initial_data))
            if np.isnan(output[f]).any():
                print f
                nan_errors.append(f)
        if nan_errors:
            print 'Error Occurred: filters return NAN', nan_errors
            return {}
        else:
            print 'DONE!'
            return output_resampled if resample else output
        
        
        
    