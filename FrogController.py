# -*- coding:utf-8 -*-
"""
Created on Feb 27, 2018

@author: Filip Lindau
"""
import threading
import time
import logging
import traceback
import Queue
from concurrent.futures import Future
from twisted.internet import reactor, defer, error
from twisted.internet.protocol import Protocol, ClientFactory, Factory
from twisted.python.failure import Failure, reflect
import PyTango as tango
import PyTango.futures as tangof
import TangoTwisted
from TangoTwisted import TangoAttributeFactory, TangoAttributeProtocol, \
    LoopingCall, DeferredCondition, ClockReactorless, defer_later
import FrogState as fs
import numpy as np
from scipy.signal import medfilt2d
from scipy.interpolate import interp1d

import FrogCalculationSimpleGP as FrogCalculation

# reload(fs)
reload(TangoTwisted)


logger = logging.getLogger("FrogController")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

# f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class FrogController(object):
    def __init__(self, spectrometer_name, motor_name, start=False):
        """
        Controller for running a scanning Frog device. Communicates with a spectrometer and a motor.


        :param spectrometer_name:
        :param motor_name:
        """
        self.device_names = dict()
        self.device_names["spectrometer"] = spectrometer_name
        self.device_names["motor"] = motor_name

        self.device_factory_dict = dict()

        self.setup_attr_params = dict()
        self.setup_attr_params["speed"] = ("motor", "speed", 50.0)
        self.setup_attr_params["acceleration"] = ("motor", "acceleration", 36.0)
        self.setup_attr_params["step_per_unit"] = ("motor", "step_per_unit", 324.0)
        self.setup_attr_params["wavelengths"] = ("spectrometer", "wavelengthvector", None)

        self.idle_params = dict()
        self.idle_params["scan_interval"] = 5.0
        self.idle_params["paused"] = False

        self.scan_params = dict()
        self.scan_params["start_pos"] = 8.6
        self.scan_params["step_size"] = 0.002
        self.scan_params["end_pos"] = 8.75
        self.scan_params["average"] = 1
        self.scan_params["scan_attr"] = "position"
        self.scan_params["spectrum_frametime"] = 0.1
        # self.scan_params["dev_name"] = "motor"

        self.scan_result = dict()
        self.scan_result["pos_data"] = None
        self.scan_result["scan_data"] = None
        self.scan_result["start_time"] = None
        self.scan_raw_data = None
        self.scan_proc_data = None
        self.scan_roi_data = None
        self.time_roi_data = None
        self.wavelength_roi_data = None

        self.analysis_result = dict()
        self.analysis_result["delta_t"] = None
        self.analysis_result["delta_ph"] = None
        self.analysis_result["error"] = None
        self.analysis_result["E_t"] = None
        self.analysis_result["ph_t"] = None
        self.analysis_result["t"] = None
        self.analysis_result["frog_rec_image"] = None

        self.wavelength_vector = None
        self.frog_calc = FrogCalculation.FrogCalculation()

        self.analyse_params = dict()
        self.analyse_params["method"] = "GP"
        self.analyse_params["algo"] = "SHG"
        self.analyse_params["size"] = 128
        self.analyse_params["dt"] = None
        self.analyse_params["iterations"] = 70
        self.analyse_params["roi"] = "full"
        self.analyse_params["threshold"] = 0.02
        self.analyse_params["median_kernel"] = 1
        self.analyse_params["background_subtract"] = True

        self.logger = logging.getLogger("FrogController.Controller")
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("FrogController.__init__")

        self.state_lock = threading.Lock()
        self.status = ""
        self.state = "unknown"
        self.state_notifier_list = list()       # Methods in this list will be called when the state
        # or status message is changed

        if start is True:
            self.device_factory_dict["spectrometer"] = TangoAttributeFactory(spectrometer_name)
            self.device_factory_dict["motor"] = TangoAttributeFactory(motor_name)

            for dev_fact in self.device_factory_dict:
                self.device_factory_dict[dev_fact].startFactory()

    def read_attribute(self, name, device_name):
        self.logger.info("Read attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[device_name]]
            d = factory.buildProtocol("read", name)
        else:
            self.logger.error("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
            err = tango.DevError(reason="Device {0} not used".format(device_name),
                                 severety=tango.ErrSeverity.ERR,
                                 desc="The device is not in the list of devices used by this controller",
                                 origin="read_attribute")
            d = Failure(tango.DevFailed(err))
        return d

    def write_attribute(self, name, device_name, data):
        self.logger.info("Write attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[device_name]]
            d = factory.buildProtocol("write", name, data)
        else:
            self.logger.error("Device name {0} not found among {1}".format(device_name, self.device_factory_dict))
            err = tango.DevError(reason="Device {0} not used".format(device_name),
                                 severety=tango.ErrSeverity.ERR,
                                 desc="The device is not in the list of devices used by this controller",
                                 origin="write_attribute")
            d = Failure(tango.DevFailed(err))
        return d

    def defer_later(self, delay, delayed_callable, *a, **kw):
        self.logger.info("Calling {0} in {1} seconds".format(delayed_callable, delay))

        def defer_later_cancel(deferred):
            delayed_call.cancel()

        d = defer.Deferred(defer_later_cancel)
        d.addCallback(lambda ignored: delayed_callable(*a, **kw))
        delayed_call = threading.Timer(delay, d.callback, [None])
        delayed_call.start()
        return d

    def check_attribute(self, attr_name, dev_name, target_value, period=0.3, timeout=1.0, tolerance=None, write=True):
        """
        Check an attribute to see if it reaches a target value. Returns a deferred for the result of the
        check.
        Upon calling the function the target is written to the attribute if the "write" parameter is True.
        Then reading the attribute is polled with the period "period" for a maximum number of retries.
        If the read value is within tolerance, the callback deferred is fired.
        If the read value is outside tolerance after retires attempts, the errback is fired.
        The maximum time to check is then period x retries

        :param attr_name: Tango name of the attribute to check, e.g. "position"
        :param dev_name: Tango device name to use, e.g. "gunlaser/motors/zaber01"
        :param target_value: Attribute value to wait for
        :param period: Polling period when checking the value
        :param timeout: Time to wait for the attribute to reach target value
        :param tolerance: Absolute tolerance for the value to be accepted
        :param write: Set to True if the target value should be written initially
        :return: Deferred that will fire depending on the result of the check
        """
        self.logger.info("Check attribute \"{0}\" on \"{1}\"".format(attr_name, dev_name))
        if dev_name in self.device_names:
            factory = self.device_factory_dict[self.device_names[dev_name]]
            d = factory.buildProtocol("check", attr_name, None, write=write, target_value=target_value,
                                      tolerance=tolerance, period=period, timeout=timeout)
        else:
            self.logger.error("Device name {0} not found among {1}".format(dev_name, self.device_factory_dict))
            err = tango.DevError(reason="Device {0} not used".format(dev_name),
                                 severety=tango.ErrSeverity.ERR,
                                 desc="The device is not in the list of devices used by this controller",
                                 origin="write_attribute")
            d = Failure(tango.DevFailed(err))
        return d

    def get_state(self):
        with self.state_lock:
            st = self.state
        return st

    def set_state(self, state):
        with self.state_lock:
            self.state = state
            for m in self.state_notifier_list:
                m(self.state, self.status)

    def get_status(self):
        with self.state_lock:
            st = self.status
        return st

    def set_status(self, status_msg):
        self.logger.debug("Status: {0}".format(status_msg))
        with self.state_lock:
            self.status = status_msg
            for m in self.state_notifier_list:
                m(self.state, self.status)

    def add_state_notifier(self, state_notifier_method):
        self.state_notifier_list.append(state_notifier_method)

    def remove_state_notifier(self, state_notifier_method):
        try:
            self.state_notifier_list.remove(state_notifier_method)
        except ValueError:
            self.logger.warning("Method {0} not in list. Ignoring.".format(state_notifier_method))

    def cond_frog_inv(self):
        I_data = np.fft.fftshift(self.frog_calc.get_reconstructed_intensity(), axes=1)
        w_vect = self.frog_calc.get_w() + self.frog_calc.w0
        l_vect = 2 * np.pi * 299792458.0 / w_vect
        t_vect = self.frog_calc.get_t()
        l_samp = self.wavelength_roi_data
        t_samp = self.time_roi_data - self.time_roi_data.mean()
        I_l_tmp = np.zeros((I_data.shape[1], l_samp.shape[0]))
        I_l_t = np.zeros((t_samp.shape[0], l_samp.shape[0]))
        for t in range(I_data.shape[0]):
            ip = interp1d(l_vect, I_data[t, :], kind="linear", fill_value=0.0, bounds_error=False)
            I_l_tmp[t, :] = ip(l_samp)
        for l in range(I_l_tmp.shape[1]):
            ip = interp1d(t_vect, I_l_tmp[:, l], kind="linear", fill_value=0.0, bounds_error=False)
            I_l_t[:, l] = ip(t_samp)
        return I_l_t


class Scan(object):
    def __init__(self, controller, scan_attr_name, scan_dev_name, start_pos, stop_pos, step,
                 meas_attr_name, meas_dev_name):
        self.controller = controller    # type: FrogController.FrogController
        self.scan_attr = scan_attr_name
        self.scan_dev = scan_dev_name
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.step = step
        self.meas_attr = meas_attr_name
        self.meas_dev = meas_dev_name

        self.current_pos = None
        self.pos_data = []
        self.meas_data = []
        self.scan_start_time = None
        self.scan_arrive_time = time.time()
        self.spectrum_time = time.time()
        self.status_update_time = time.time()
        self.status_update_interval = 1.0

        self.logger = logging.getLogger("FrogController.Scan_{0}_{1}".format(self.scan_attr, self.meas_attr))
        self.logger.setLevel(logging.DEBUG)

        self.d = defer.Deferred()
        self.cancel_flag = False

        # Add errback handling!

    def start_scan(self):
        self.logger.info("Starting scan of {0} from {1} to {2} measuring {3}".format(self.scan_attr.upper(),
                                                                                     self.start_pos,
                                                                                     self.stop_pos,
                                                                                     self.meas_attr.upper()))
        self.scan_start_time = time.time()
        self.spectrum_time = time.time()
        self.status_update_time = self.scan_start_time
        self.cancel_flag = False
        scan_pos = self.start_pos
        tol = self.step * 0.1

        d0 = self.controller.check_attribute(self.scan_attr, self.scan_dev, scan_pos,
                                             0.1, 10.0, tolerance=tol, write=True)
        d0.addCallbacks(self.scan_arrive, self.scan_error_cb)
        # d0.addCallback(lambda _: self.controller.read_attribute(self.meas_attr, self.meas_dev))
        # d0.addCallback(self.meas_scan)
        self.d = defer.Deferred(self.cancel_scan)

        return self.d

    def scan_step(self):
        self.logger.info("Scan step")
        tol = self.step * 0.1
        scan_pos = self.current_pos + self.step
        if scan_pos > self.stop_pos or self.cancel_flag is True:
            self.scan_done()
            return self.d

        d0 = self.controller.check_attribute(self.scan_attr, self.scan_dev, scan_pos,
                                             0.1, 3.0, tolerance=tol, write=True)
        d0.addCallbacks(self.scan_arrive, self.scan_error_cb)
        # d0.addCallback(lambda _: self.controller.read_attribute(self.meas_attr, self.meas_dev))
        # d0.addCallback(self.meas_scan)
        return d0

    def scan_arrive(self, result):
        try:
            self.current_pos = result.value
        except AttributeError:
            self.logger.error("Scan arrive not returning attribute: {0}".format(result))
            return False

        self.logger.info("Scan arrive at pos {0}".format(self.current_pos))
        t = time.time()
        if t - self.status_update_time > self.status_update_interval:
            status = "Scanning from {0} to {1} with step size {2}\n" \
                     "Position: {3}".format(self.start_pos, self.stop_pos, self.step, self.current_pos)
            self.controller.set_status(status)
            self.status_update_time = t

        # After arriving, check the time against the last read spectrum. Wait for a time so that the next
        # spectrum frame should be there.
        self.scan_arrive_time = t
        try:
            wait_time = (self.scan_arrive_time - self.spectrum_time) % self.controller.scan_params["spectrum_frametime"]
        except KeyError:
            wait_time = 0.1
        # print("Waittime: {0}".format(wait_time))
        d0 = defer_later(wait_time, self.meas_read)
        d0.addErrback(self.scan_error_cb)
        self.logger.debug("Scheduling read in {0} s".format(wait_time))
        # d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev)
        # d0.addCallback(test_cb2)
        # d0.addCallbacks(self.meas_scan, self.scan_error_cb)

        return True

    def meas_read(self):
        """
        Called after scan_arrive to read a new spectrum
        :return:
        """
        self.logger.info("Reading measurement")
        d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev)
        d0.addCallbacks(self.meas_scan, self.scan_error_cb)
        return True

    def meas_scan(self, result):
        self.logger.debug("Meas scan result: {0}".format(result.value))
        if result.time.totime() <= self.scan_arrive_time:
            self.logger.debug("Old spectrum. Wait for new.")
            t = time.time() - result.time.totime()
            if t > 2.0:
                self.logger.error("Timeout waiting for new spectrum. {0} s elapsed".format(t))
                self.scan_error_cb(RuntimeError("Timeout waiting for new spectrum"))
                return False
            d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev)
            d0.addCallbacks(self.meas_scan, self.scan_error_cb)
            return False
        self.spectrum_time = result.time.totime()
        measure_value = result.value
        # try:
        #     measure_value = result.value
        # except AttributeError:
        #     self.logger.error("Measurement not returning attribute: {0}".format(result))
        #     return False
        self.logger.debug("Measure at scan pos {0} result: {1}".format(self.current_pos, measure_value))
        self.meas_data.append(measure_value)
        self.pos_data.append(self.current_pos)
        self.scan_step()
        return True

    def scan_done(self):
        self.logger.info("Scan done!")
        scan_raw_data = np.array(self.meas_data)
        self.logger.info("Scan dimensions: {0}".format(scan_raw_data.shape))
        pos_data = np.array(self.pos_data)
        self.d.callback([pos_data, scan_raw_data, self.scan_start_time])

    def scan_error_cb(self, err):
        self.logger.error("Scan error: {0}".format(err))
        # Here we can handle the error if it is manageable or
        # propagate the error to the calling callback chain:
        if err.type in []:
            pass
        else:
            self.d.errback(err)

    def cancel_scan(self, result):
        self.logger.info("Cancelling scan, result {0}".format(result))
        self.cancel_flag = True


class FrogAnalyse(object):
    """
    Steps to take during analysis:

    1. Load data
    2. Pre-process data... threshold, background subtract, filter
    3. Convert to t-f space
    4. Run FROG algo

    """
    def __init__(self, controller):

        self.controller = controller    # type: FrogController.FrogController

        self.scan_raw_data = None
        self.scan_proc_data = None
        self.scan_roi_data = None
        self.time_data = None
        self.wavelength_data = None
        self.time_roi_data = None
        self.wavelength_roi_data = None
        self.frog_rec_image = None

        self.logger = logging.getLogger("FrogController.Analysis")
        self.logger.setLevel(logging.DEBUG)

        self.d = defer.Deferred()

    def start_analysis(self):
        self.logger.info("Starting up frog analysis")
        self.load_data()
        self.preprocess()
        self.find_roi()
        self.convert_data(use_roi=True)
        d = TangoTwisted.defer_to_thread(self.invert_frog_trace)
        d.addCallbacks(self.retrieve_data, self.analysis_error)
        self.d = defer.Deferred()
        return self.d

    def load_data(self):
        self.logger.info("Loading frog data from scan")
        scan_result = self.controller.scan_result
        pos = np.array(scan_result["pos_data"])  # Vector containing the motor positions during the scan
        t = (pos-pos[0]) * 2 * 1e-3 / 299792458.0   # Assume position is in mm
        # The wavelengths should have been read earlier.
        if self.controller.wavelength_vector is None:
            self.logger.error("Wavelength vector not read")
            fail = Failure(AttributeError("Wavelength vector not read"))
            self.d.errback(fail)
            return
        w = self.controller.wavelength_vector
        self.time_data = t
        self.wavelength_data = w
        self.scan_raw_data = np.array(scan_result["scan_data"])
        if self.time_data.shape[0] != self.scan_raw_data.shape[0]:
            err_s = "Time vector not matching scan_data dimension: {0} vs {1}".format(self.time_data.shape[0],
                                                                                      self.scan_raw_data.shape[0])
            self.logger.error(err_s)
            fail = Failure(AttributeError(err_s))
            self.d.errback(fail)
            return
        if w.shape[0] != self.scan_raw_data.shape[1]:
            err_s = "Wavelength vector not matching scan_data dimension: {0} vs {1}".format(w.shape[0],
                                                                                            self.scan_raw_data.shape[0])
            self.logger.error(err_s)
            fail = Failure(AttributeError(err_s))
            self.d.errback(fail)
            return

    def preprocess(self):
        """
        Preprocess data to improve the FROG inversion quality.
        The most important part is the thresholding to isolate the data part
        of the FROG trace.
        We also do background subtraction, normalization, and filtering.
        Thresholding is done after background subtraction.
        :return:
        """
        self.logger.info("Preprocessing scan data")
        if self.controller.analyse_params["background_subtract"] is True:
            # Use first and last spectrum lines as background level.
            # We should start and end the scan outside the trace anyway.
            bkg0 = self.scan_raw_data[0, :]
            bkg1 = self.scan_raw_data[-1, :]
            bkg = (bkg0 + bkg1) / 2
            # Tile background vector to a 2D matrix that can be subtracted from the data:
            bkg_m = np.tile(bkg, (self.scan_raw_data.shape[0], 1))
            proc_data = self.scan_raw_data - bkg_m
            self.logger.debug("Background image subtractged")
        else:
            proc_data = np.copy(self.scan_raw_data)
        # Normalization
        proc_data = proc_data / proc_data.max()
        self.logger.debug("Scan data normalized")
        # Thresholding
        thr = self.controller.analyse_params["threshold"]
        self.logger.debug("Threshold: {0}".format(thr))
        proc_thr = np.clip(proc_data - thr, 0, None)     # Thresholding
        self.logger.debug("Scan data thresholded to {0}".format(thr))
        # Filtering
        kernel = int(self.controller.analyse_params["median_kernel"])
        if kernel > 1:
            proc_thr = medfilt2d(proc_thr, kernel)
            self.logger.debug("Scan data median filtered with kernel size {0}".format(kernel))
        # self.logger.debug("proc_data {0}".format(proc_data))
        # self.logger.debug("proc_thr {0}".format(proc_thr))
        self.scan_proc_data = proc_thr

    def find_roi(self):
        """
        Find the ROI around the centroid of the processed scan image.
        :return:
        """
        self.logger.info("Running find_roi to center frog trace around data")
        I_tot = self.scan_proc_data.sum()
        xr = np.arange(self.scan_proc_data.shape[0])
        x_cent = (xr * self.scan_proc_data.sum(1)).sum() / I_tot
        yr = np.arange(self.scan_proc_data.shape[1])
        y_cent = (yr * self.scan_proc_data.sum(0)).sum() / I_tot
        self.logger.debug("Centroid position: {0:.1f} x {1:.1f}".format(x_cent, y_cent))
        xw = np.floor(np.minimum(x_cent - xr[0], xr[-1] - x_cent)).astype(np.int)
        yw = np.floor(np.minimum(y_cent - yr[0], yr[-1] - y_cent)).astype(np.int)
        x0 = np.floor(x_cent - xw).astype(np.int)
        x1 = np.floor(x_cent + xw).astype(np.int)
        y0 = np.floor(y_cent - yw).astype(np.int)
        y1 = np.floor(y_cent + yw).astype(np.int)
        self.scan_roi_data = self.scan_proc_data[x0:x1, y0:y1]
        self.time_roi_data = self.time_data[x0:x1]
        self.wavelength_roi_data = self.wavelength_data[y0:y1]
        self.controller.scan_raw_data = self.scan_raw_data
        self.controller.scan_proc_data = self.scan_proc_data
        self.controller.scan_roi_data = self.scan_roi_data
        self.controller.wavelength_roi_data = self.wavelength_roi_data
        self.controller.time_roi_data = self.time_roi_data

    def convert_data(self, use_roi=False):
        """
        Create the reference intensity frog trace for the algorithm.
        We need the wavelength range and the time range.
        Also the size of the transform is required.
        :return:
        """
        self.logger.info("Converting scan data for FROG algorithm")
        if use_roi is False:
            tau_mean = (self.time_data[-1] + self.time_data[0]) / 2
            tau_start = self.time_data[0] - tau_mean
            tau_stop = self.time_data[-1] - tau_mean
            l_start = self.wavelength_data[0]
            l_stop = self.wavelength_data[-1]
            data = self.scan_proc_data
        else:
            tau_mean = (self.time_roi_data[-1] + self.time_roi_data[0]) / 2
            tau_start = self.time_roi_data[0] - tau_mean
            tau_stop = self.time_roi_data[-1] - tau_mean
            l_start = self.wavelength_roi_data[0]
            l_stop = self.wavelength_roi_data[-1]
            data = self.scan_roi_data
        l0 = (l_stop + l_start) / 2
        n = self.controller.analyse_params["size"]
        dt_t = np.abs(tau_stop - tau_start) / n
        dt_l = np.abs(1.0 / (1.0 / l_start - 1.0 / l_stop) / 299792458.0)
        self.logger.debug("dt_t: {0}, dt_l: {1}".format(dt_t, dt_l))
        dt_p = self.controller.analyse_params["dt"]
        if dt_p is None:
            dt = (dt_t + dt_l) / 2
        else:
            dt = dt_p
        self.controller.frog_calc.init_pulsefield_random(n, dt, l0)
        self.logger.debug("Pulse field initialized with n={0}, dt={1} fs, l0={2} nm".format(n, dt*1e15, l0*1e9))
        self.controller.frog_calc.condition_frog_trace2(data, l_start, l_stop,
                                                        tau_start, tau_stop, n, thr=0)
        self.logger.debug("Frog trace conditioned to t-f space")

    def invert_frog_trace(self):
        self.logger.info("Invert FROG trace")
        method = self.controller.analyse_params["method"]
        iterations = self.controller.analyse_params["iterations"]
        algo = self.controller.analyse_params["algo"].upper()
        if algo not in ["SHG", "SD", "PG"]:
            err_str = "Algorithm {0} not recognized. Should be SHG, SD, or PG.".format(algo)
            self.logger.error(err_str)
            e = AttributeError(err_str)
            self.d.errback(Failure(e))
        self.logger.info("Inverting {0} FROG trace using {1} with {2} iterations.".format(algo,
                                                                                          method,
                                                                                          iterations))
        if method.lower() == "gp":
            self.controller.frog_calc.run_cycle_gp(iterations, algo)
        else:
            self.controller.frog_calc.run_cycle_vanilla(iterations, algo)

    def cond_frog_inv(self):
        I_data = np.fft.fftshift(self.controller.frog_calc.get_reconstructed_intensity(), axes=1)
        w_vect = self.controller.frog_calc.get_w() + self.controller.frog_calc.w0
        l_vect = 2 * np.pi * 299792458.0 / w_vect
        t_vect = self.controller.frog_calc.get_t()
        l_samp = self.wavelength_roi_data
        t_samp = self.time_roi_data - self.time_roi_data.mean()
        I_l_tmp = np.zeros((I_data.shape[1], l_samp.shape[0]))
        I_l_t = np.zeros((t_samp.shape[0], l_samp.shape[0]))
        for t in range(I_data.shape[0]):
            ip = interp1d(l_vect, I_data[t, :], kind="linear", fill_value=0.0, bounds_error=False)
            I_l_tmp[t, :] = ip(l_samp)
        for l in range(I_l_tmp.shape[1]):
            ip = interp1d(t_vect, I_l_tmp[:, l], kind="linear", fill_value=0.0, bounds_error=False)
            I_l_t[:, l] = ip(t_samp)
        self.frog_rec_image = I_l_t

    def retrieve_data(self, result):
        """
        Retrieve data from frog analysis run. Run as a callback from a deferred.
        :param result: deferred result
        :return:
        """
        self.logger.info("Retrieving data from reconstruction")
        frog_calc = self.controller.frog_calc
        delta_t, delta_ph = frog_calc.get_trace_summary(domain="temporal")
        rec_err = frog_calc.G_hist[-1]
        E_t = frog_calc.get_trace_abs()
        ph_t = frog_calc.get_trace_phase()
        t = frog_calc.get_t()
        try:
            dt = t[1] - t[0]
        except (TypeError, IndexError):
            dt = None

        self.controller.analysis_result["delta_t"] = delta_t
        self.controller.analysis_result["delta_ph"] = delta_ph
        self.controller.analysis_result["error"] = rec_err
        self.controller.analysis_result["E_t"] = E_t
        self.controller.analysis_result["ph_t"] = ph_t
        self.controller.analysis_result["t"] = t
        self.controller.analysis_result["dt"] = dt
        self.cond_frog_inv()
        self.controller.analysis_result["frog_rec_image"] = self.frog_rec_image
        self.d.callback(self.controller.analysis_result)

    def analysis_error(self, err):
        self.logger.error("Error in FROG analysis: {0}".format(err))
        self.d.errback(err)


def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))


def test_timeout(result):
    logger.warning("TIMEOUT returned {0}".format(result))


if __name__ == "__main__":
    # fc = FrogController("sys/tg_test/1", "gunlaser/motors/zaber01")
    fc = FrogController("gunlaser/devices/spectrometer_frog", "gunlaser/motors/zaber01")
    # time.sleep(0)
    # dc = fc.check_attribute("position", "motor", 7.17, 0.1, 0.5, 0.001, True)
    # dc.addCallbacks(test_cb, test_err)
    # time.sleep(1.0)
    # dc.addCallback(lambda _: TangoTwisted.DelayedCallReactorless(2.0 + time.time(),
    #                                                              fc.start_scan, ["position", 5, 10, 0.5,
    #                                                                              "double_scalar"]))
    # scan = TangoTwisted.Scan(fc, "position", "motor", 5, 10, 0.5, "double_scalar", "spectrometer")
    # ds = scan.start_scan()
    # ds.addCallback(test_cb)

    sh = fs.FrogStateDispatcher(fc)
    sh.start()

    # da = fc.read_attribute("double_scalar", "motor")
    # da.addCallbacks(test_cb, test_err)
    # da = fc.write_attribute("double_scalar_w", "motor", 10)
    # da.addCallbacks(test_cb, test_err)

    # da = fc.defer_later(3.0, fc.read_attribute, "short_scalar", "motor")
    # da.addCallback(test_cb, test_err)

    # lc = LoopingCall(fc.read_attribute, "double_scalar_w", "motor")
    # dlc = lc.start(1)
    # dlc.addCallbacks(test_cb, test_err)
    # lc.loop_deferred.addCallback(test_cb)

    # clock = ClockReactorless()
    #
    # defcond = DeferredCondition("result.value>15", fc.read_attribute, "double_scalar_w", "motor")
    # dcd = defcond.start(1.0, timeout=3.0)
    # dcd.addCallbacks(test_cb, test_err)

