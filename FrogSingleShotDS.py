# -*- coding:utf-8 -*-
"""
Created on Apr 11, 2018

@author: Filip Lindau
"""
import time

import PyTango as pt
import numpy as np
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command
from PyTango.server import device_property

from FrogControllerSingleShot import FrogController
from FrogStateSingleShot import FrogStateDispatcher


class FrogSingleShotDS(Device):
    __metaclass__ = DeviceMeta

    # --- Expert attributes
    #
    dt = attribute(label='dt',
                   dtype=float,
                   access=pt.AttrWriteType.READ,
                   display_level=pt.DispLevel.EXPERT,
                   unit="s",
                   format="%3.2e",
                   min_value=0.0,
                   max_value=1.0,
                   fget="get_dt",
                   doc="FROG trace time resolution", )

    # --- Operator attributes
    #
    delta_t = attribute(label='delta_t',
                        dtype=float,
                        access=pt.AttrWriteType.READ,
                        unit="s",
                        format="%3.2e",
                        min_value=0.0,
                        max_value=1.0,
                        fget="get_delta_t",
                        doc="Reconstructed pulse time intensity FWHM",)

    rec_error = attribute(label='reconstruction error',
                          dtype=float,
                          access=pt.AttrWriteType.READ,
                          unit="rel",
                          format="%3.3f",
                          min_value=0.0,
                          max_value=1.0,
                          fget="get_rec_error",
                          doc="Relative reconstruction error",)

    timevector = attribute(label='TimeVector',
                           dtype=[np.double],
                           access=pt.AttrWriteType.READ,
                           max_dim_x=16384,
                           display_level=pt.DispLevel.OPERATOR,
                           unit="m",
                           format="%5.2e",
                           fget="get_timevector",
                           doc="Time vector for FROG trace",
                           )

    e_field = attribute(label='ElectricField',
                        dtype=[np.double],
                        access=pt.AttrWriteType.READ,
                        max_dim_x=16384,
                        display_level=pt.DispLevel.OPERATOR,
                        unit="a.u.",
                        format="%5.2f",
                        fget="get_efield",
                        doc="Reconstructed electric field vs time",
                        )

    phase = attribute(label='Phase',
                      dtype=[np.double],
                      access=pt.AttrWriteType.READ,
                      max_dim_x=16384,
                      display_level=pt.DispLevel.OPERATOR,
                      unit="rad",
                      format="%5.2f",
                      fget="get_phase",
                      doc="Reconstructed phase vs time",
                      )

    scan_raw_data = attribute(label='Scan raw data',
                              dtype=[[np.double]],
                              access=pt.AttrWriteType.READ,
                              max_dim_x=2048,
                              max_dim_y=2048,
                              display_level=pt.DispLevel.OPERATOR,
                              unit="a.u.",
                              format="%5.2f",
                              fget="get_scan_data",
                              doc="Latest frog trace scan raw data",
                              )

    frog_in_image = attribute(label='FROG input image',
                              dtype=[[np.double]],
                              access=pt.AttrWriteType.READ,
                              max_dim_x=2048,
                              max_dim_y=2048,
                              display_level=pt.DispLevel.OPERATOR,
                              unit="a.u.",
                              format="%5.2f",
                              fget="get_frog_in_image",
                              doc="Latest frog trace conditioned input image",
                              )

    frog_rec_image = attribute(label='FROG reconstructed image',
                               dtype=[[np.double]],
                               access=pt.AttrWriteType.READ,
                               max_dim_x=2048,
                               max_dim_y=2048,
                               display_level=pt.DispLevel.OPERATOR,
                               unit="a.u.",
                               format="%5.2f",
                               fget="get_frog_rec_image",
                               doc="Latest frog trace reconstructed image",
                               )

    camera_name = device_property(dtype=str,
                                  doc="Tango name of the camera device",
                                  default_value="gunlaser/cameras/spectrometer_camera")

    camera_time_res = device_property(dtype=float,
                                      doc="Time resolution of camera image in fs/pixel",
                                      default_value=1.5)

    camera_spectral_res = device_property(dtype=float,
                                          doc="Spectral resolution of camera image in nm/pixel. Use negative number "
                                              "to indicate wavelength decreasing for higher pixel numbers.",
                                          default_value=-0.0585)

    camera_central_wavelength = device_property(dtype=float,
                                                doc="Central wavelength of camera image for the spectral "
                                                    "axis (horizontal) in nm.",
                                                default_value=388)

    scan_interval = device_property(dtype=np.double,
                                    doc="Time interval between FROG scans in s",
                                    default_value=10.0)

    scan_average_number = device_property(dtype=int,
                                          doc="Number of images to average to produce a FROG trace",
                                          default_value=11)

    analysis_method = device_property(dtype=str,
                                      doc="FROG method to use for analysis. gp or vanilla\n"
                                          "(general projections or vanilla, where gp is normally best)",
                                      default_value="gp")

    analysis_algo = device_property(dtype=str,
                                    doc="FROG algorithm to use for analysis:\n"
                                        "shg (second harmonic generation),\n"
                                        "sd (self diffraction), or\n"
                                        "pg (polarization gating)",
                                    default_value="shg")

    analysis_size = device_property(dtype=np.int,
                                    doc="Size of FROG inversion in pixels",
                                    default_value=128)

    analysis_iterations = device_property(dtype=np.int,
                                          doc="Number of iterations for FROG inversion",
                                          default_value=70)

    analysis_background = device_property(dtype=np.bool,
                                          doc="Boolean if the image should be background subtracted "
                                              "using the first position",
                                          default_value=True)

    analysis_threshold = device_property(dtype=np.double,
                                         doc="Threshold level for normalized (and bkg subtracted) image",
                                         default_value=0.02)

    analysis_median = device_property(dtype=np.int,
                                      doc="Kernel size for median filtering the raw FROG data.\n"
                                          "If set to 1 there is no median filtering.\n"
                                          "Must be an odd number",
                                      default_value=1)

    def __init__(self, klass, name):
        self.max_value = 1.0
        self.controller = None              # type: FrogController
        self.setup_attr_params = dict()
        self.idle_params = dict()
        self.scan_params = dict()
        self.analyse_params = dict()
        self.db = None
        self.frogstate_dispatcher = None    # type: FrogStateDispatcher
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.db = pt.Database()
        self.set_state(pt.DevState.UNKNOWN)
        try:
            if self.frogstate_dispatcher is not None:
                self.frogstate_dispatcher.stop()
        except Exception as e:
            self.error_info("Error stopping state dispatcher: {0}".format(e))
        try:
            self.controller = FrogController(self.camera_name)
            self.controller.add_state_notifier(self.change_state)
        except Exception as e:
            self.error_stream("Error creating camera controller: {0}".format(e))
            return

        self.setup_params()

        self.frogstate_dispatcher = FrogStateDispatcher(self.controller)
        self.frogstate_dispatcher.start()

        self.debug_stream("init_device finished")
        # self.set_state(pt.DevState.ON)

    def setup_params(self):
        # Populate parameter dicts from device properties
        self.setup_attr_params = dict()
        # self.setup_attr_params["wavelengths"] = ("spectrometer", "wavelengthvector", None)
        self.debug_stream("setup_attr_params: {0}".format(self.setup_attr_params))

        self.idle_params = dict()
        self.idle_params["scan_interval"] = self.scan_interval
        self.idle_params["paused"] = False
        self.debug_stream("idle_params: {0}".format(self.idle_params))

        self.scan_params = dict()
        self.scan_params["image_attr"] = "image"
        self.scan_params["average"] = self.scan_average_number
        self.scan_params["time_res"] = self.camera_time_res * 1e-15
        self.scan_params["spectral_res"] = self.camera_spectral_res * 1e-9
        self.scan_params["central_wavelength"] = self.camera_central_wavelength * 1e-9
        self.debug_stream("scan_params: {0}".format(self.scan_params))

        self.analyse_params = dict()
        self.analyse_params["method"] = self.analysis_method
        self.analyse_params["algo"] = self.analysis_algo
        self.analyse_params["size"] = self.analysis_size
        self.analyse_params["dt"] = None
        self.analyse_params["iterations"] = self.analysis_iterations
        self.analyse_params["roi"] = "full"
        self.analyse_params["threshold"] = self.analysis_threshold
        self.analyse_params["median_kernel"] = self.analysis_median
        self.analyse_params["background_subtract"] = self.analysis_background
        self.debug_stream("analyse_params: {0}".format(self.analyse_params))

        # Copy these to frog controller
        self.controller.setup_attr_params = self.setup_attr_params
        self.controller.idle_params = self.idle_params
        self.controller.scan_params = self.scan_params
        self.controller.analyse_params = self.analyse_params

    def change_state(self, new_state, new_status=None):
        self.info_stream("Change state: {0}, status {1}".format(new_state, new_status))
        # Map new_state string to tango state
        if new_state in ["analyse"]:
            tango_state = pt.DevState.RUNNING
        elif new_state in ["scan"]:
            tango_state = pt.DevState.MOVING
        elif new_state in ["idle"]:
            tango_state = pt.DevState.ON
        elif new_state in ["device_connect", "setup_attributes"]:
            tango_state = pt.DevState.INIT
        elif new_state in ["fault"]:
            tango_state = pt.DevState.FAULT
        else:
            tango_state = pt.DevState.UNKNOWN
        if tango_state != self.get_state():
            self.debug_stream("Change state from {0} to {1}".format(self.get_state(), new_state))
            self.set_state(tango_state)
        if new_status is not None:
            self.debug_stream("Setting status {0}".format(new_status))
            self.set_status(new_status)

    def get_timevector(self):
        value = self.controller.analysis_result["t"]
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_efield(self):
        value = self.controller.analysis_result["E_t"]
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_phase(self):
        value = self.controller.analysis_result["ph_t"]
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_delta_t(self):
        value = self.controller.analysis_result["delta_t"]
        t = self.controller.scan_result["start_time"]
        self.info_stream("In get_delta_t:value={0}, start time={1}".format(value, t))
        self.debug_stream("Time type: {0}".format(type(t)))
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_dt(self):
        value = self.controller.analysis_result["dt"]
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_rec_error(self):
        value = self.controller.analysis_result["error"]
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_scan_data(self):
        value = self.controller.scan_raw_data
        self.debug_stream("Raw scan data dimension: {0}".format(value.shape))
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_frog_in_image(self):
        value = self.controller.scan_roi_data
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_frog_rec_image(self):
        self.info_stream("Get FROG rec image called")
        value = self.controller.analysis_result["frog_rec_image"]
        self.debug_stream("Reconstructed FROG image dimensions: {0}".format(value.shape))
        t = self.controller.scan_result["start_time"]
        if t is None:
            t = time.time()
        if isinstance(t, pt.TimeVal):
            t = t.totime()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    @command(doc_in="Pause scanning operations")
    def pause_scanning(self):
        """Pause scanning of frog traces"""
        self.info_stream("Pausing scanning operations")
        self.frogstate_dispatcher.send_command("pause")

    @command(doc_in="Start scanning operations.")
    def do_scan(self):
        """Resume scanning of frog traces"""
        self.info_stream("Resuming scanning operations")
        self.frogstate_dispatcher.send_command("scan")

    @command(doc_in="Do reconstruction with current scan data.")
    def do_reconstruction(self):
        """Run reconstruction on current scan data"""
        self.info_stream("Running FROG reconstruction")
        self.frogstate_dispatcher.send_command("analyse")

    @command(doc_in="Re-read device properties.\n " \
                    "This can be used to alter analysis parameters with current scan data.")
    def read_properties(self):
        """Re-read device properties"""
        self.info_stream("Reading device properties")
        self.get_device_properties()
        pause = self.controller.idle_params["pause"]
        self.setup_params()
        self.controller.idle_params["pause"] = pause


if __name__ == "__main__":
    pt.server.server_run((FrogSingleShotDS,))
