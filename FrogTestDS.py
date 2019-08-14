# -*- coding:utf-8 -*-
"""
Created on Aug 11, 2018

@author: Filip Lindau
"""
import time

import PyTango as pt
import numpy as np
import copy
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute, command
from PyTango.server import device_property

from FrogControllerSingleShot import FrogController
from FrogStateSingleShot import FrogStateDispatcher


class FrogTestDS(Device):
    __metaclass__ = DeviceMeta

    im_size = attribute(label='image size',
                        dtype=int,
                        access=pt.AttrWriteType.READ_WRITE,
                        memorized=True,
                        hw_memorized=True,
                        unit="pixels",
                        format="%3d",
                        min_value=1,
                        max_value=2048,
                        fget="get_im_size",
                        fset="set_im_size",
                        doc="Image size",)

    image = attribute(label='image',
                      dtype=((np.double, ), ),
                      access=pt.AttrWriteType.READ,
                      max_dim_x=2048,
                      max_dim_y=2048,
                      display_level=pt.DispLevel.OPERATOR,
                      unit="a.u.",
                      format="%5.2f",
                      fget="get_image",
                      doc="Latest image",
                      )

    e_field = attribute(label='ElectricField',
                        dtype=(np.double, ),
                        access=pt.AttrWriteType.READ,
                        max_dim_x=16384,
                        display_level=pt.DispLevel.OPERATOR,
                        unit="a.u.",
                        format="%5.2f",
                        fget="get_efield",
                        doc="Reconstructed electric field vs time",
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
                                                default_value=388.0)

    scan_interval = device_property(dtype=np.double,
                                    doc="Time interval between FROG scans in s",
                                    default_value=1.0)

    scan_average_number = device_property(dtype=int,
                                          doc="Number of images to average to produce a FROG trace",
                                          default_value=1)

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

    def __init__(self, cl, name):
        self._im_size = 10
        self.db = None
        Device.__init__(self, cl, name)

    def init_device(self):
        Device.init_device(self)
        self.db = pt.Database()
        self.set_state(pt.DevState.UNKNOWN)
        try:
            self.frogstate_dispatcher.stop()
        except AttributeError:
            pass

        self.controller = FrogController(self.camera_name)
        self.controller.add_state_notifier(self.change_state)
        self.setup_params()
        self.frogstate_dispatcher = FrogStateDispatcher(self.controller)
        self.frogstate_dispatcher.start()

    def setup_params(self):
        # Populate parameter dicts from device properties
        self.setup_attr_params = dict()
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

    def get_image(self):
        # value = np.random.random((self._im_size, self._im_size))
        value = self.controller.get_frog_in_image()
        self.debug_stream("==== GET_IMAGE sum {0}".format(value.sum()))
        t = self.controller.get_start_time()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def get_efield(self):
        value = self.controller.get_e_field()
        t = self.controller.get_start_time()
        if value is None:
            q = pt.AttrQuality.ATTR_INVALID
        else:
            q = pt.AttrQuality.ATTR_VALID
        return value, t, q

    def set_im_size(self, value):
        self._im_size = value

    def get_im_size(self):
        return self._im_size

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
    pt.server.server_run((FrogTestDS,))
