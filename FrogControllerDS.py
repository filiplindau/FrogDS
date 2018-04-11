# -*- coding:utf-8 -*-
"""
Created on Apr 11, 2018

@author: Filip Lindau
"""
import threading
import time
import PyTango as pt
from PyTango.server import Device, DeviceMeta
from PyTango.server import attribute
from PyTango.server import device_property
from FrogController import FrogController
from FrogState import FrogStateDispatcher
import numpy as np


class FrogControllerDS(Device):
    __metaclass__ = DeviceMeta

    delta_t = attribute(label='delta_t',
                        dtype=float,
                        access=pt.AttrWriteType.READ,
                        unit="s",
                        format="%3.2e",
                        min_value=0.0,
                        max_value=1.0,
                        fget="get_delta_t",
                        doc="Reconstructed pulse time intensity FWHM",
                        memorized=True,)
    # hw_memorized=True)

    gain = attribute(label='Gain',
                     dtype=float,
                     access=pt.AttrWriteType.READ_WRITE,
                     unit="dB",
                     format="%3.2f",
                     min_value=0.0,
                     max_value=1e2,
                     fget="get_gain",
                     fset="set_gain",
                     doc="Camera gain in dB",
                     memorized=True,)
                     # hw_memorized=True)

    timevector = attribute(label='TimeVector',
                           dtype=[np.double],
                           access=pt.AttrWriteType.READ,
                           max_dim_x=16384,
                           display_level=pt.DispLevel.OPERATOR,
                           unit="m",
                           format="%5.2e",
                           fget="get_wavelengthvector",
                           doc="Time vector",
                           )

    e_field = attribute(label='ElectricField',
                        dtype=[np.double],
                        access=pt.AttrWriteType.READ,
                        max_dim_x=16384,
                        display_level=pt.DispLevel.OPERATOR,
                        unit="a.u.",
                        format="%5.2f",
                        fget="get_spectrum",
                        doc="Reconstructed electric field vs time",
                        )

    scan_data = attribute(label='ScanData',
                          dtype=[np.double, np.double],
                          access=pt.AttrWriteType.READ,
                          max_dim_x=16384,
                          max_dim_y=16384,
                          display_level=pt.DispLevel.OPERATOR,
                          unit="a.u.",
                          format="%5.2f",
                          fget="get_scan_data",
                          doc="Latest frog trace scan data",
                          )

    motor_name = device_property(dtype=str,
                                  doc="Tango name of the motor device",
                                  default_value="gunlaser/motors/zaber01")

    spectrometer_name = device_property(dtype=str,
                                  doc="Tango name of the spectrometer device",
                                  default_value="gunlaser/devices/spectrometer_frog")

    def __init__(self, klass, name):
        self.wavelengthvector_data = np.array([])
        self.max_value = 1.0
        self.controller = None
        self.db = None
        Device.__init__(self, klass, name)

    def init_device(self):
        self.debug_stream("In init_device:")
        Device.init_device(self)
        self.db = pt.Database()
        self.set_state(pt.DevState.UNKNOWN)
        self.debug_stream("Init camera controller {0}".format(self.camera_name))
        params = dict()
        params["imageoffsetx"] = self.roi[0]
        params["imageoffsety"] = self.roi[1]
        params["imagewidth"] = self.roi[2]
        params["imageheight"] = self.roi[3]
        params["triggermode"] = "Off"
        try:
            if self.controller is not None:
                self.controller.stop_thread()
        except Exception as e:
            self.error_info("Error stopping camera controller: {0}".format(e))
        try:
            self.setup_spectrometer()
            self.controller = FrogController(self.camera_name, params,
                                             self.wavelengthvector_data,
                                             self.max_value)
            # self.dev_controller = CameraDeviceController(self.camera_name, params)
        except Exception as e:
            self.error_stream("Error creating camera controller: {0}".format(e))
            return

        self.debug_stream("init_device finished")
        # self.set_state(pt.DevState.ON)
        self.controller.add_state_callback(self.change_state)

    def setup_spectrometer(self):
        self.info_stream("Entering setup_camera")
        self.wavelengthvector_data = (self.central_wavelength + np.arange(-self.roi[2] / 2,
                                                                          self.roi[2] / 2) * self.dispersion) * 1e-9
        self.max_value = self.saturation_level

    def change_state(self, new_state, new_status=None):
        self.debug_stream("Change state from {0} to {1}".format(self.get_state(), new_state))
        if self.get_state() is pt.DevState.INIT and new_state is not pt.DevState.UNKNOWN:
            self.debug_stream("Set memorized attributes")
            data = self.db.get_device_attribute_property(self.get_name(), "gain")
            self.debug_stream("Database returned data for \"gain\": {0}".format(data["gain"]))
            try:
                new_value = float(data["gain"]["__value"][0])
                self.debug_stream("{0}".format(new_value))
                self.controller.write_attribute("gain", new_value)
            except (KeyError, TypeError, IndexError, ValueError):
                pass
            data = self.db.get_device_attribute_property(self.get_name(), "exposuretime")
            self.debug_stream("Database returned data for \"exposuretime\": {0}".format(data["exposuretime"]))
            try:
                new_value = float(data["exposuretime"]["__value"][0])
                self.controller.write_attribute("exposuretime", new_value)
            except (KeyError, TypeError, IndexError, ValueError):
                pass
        self.set_state(new_state)
        if new_status is not None:
            self.debug_stream("Setting status {0}".format(new_status))
            self.set_status(new_status)

    def get_spectrum(self):
        attr = self.controller.get_attribute("image")
        try:
            spectrum = attr.value.sum(0)
        except AttributeError:
            spectrum = []
        return spectrum, attr.time.totime(), attr.quality

    def get_wavelengthvector(self):
        self.debug_stream("get_wavelengthvector: size {0}".format(self.wavelengthvector_data.shape))
        return self.wavelengthvector_data, time.time(), pt.AttrQuality.ATTR_VALID

    def get_exposuretime(self):
        attr = self.controller.get_attribute("exposuretime")
        return attr.value, attr.time.totime(), attr.quality

    def set_exposuretime(self, new_exposuretime):
        self.debug_stream("In set_exposuretime: New value {0}".format(new_exposuretime))
        self.debug_stream("Type dev_controller: {0}".format(type(self.controller)))
        self.controller.write_attribute("exposuretime", new_exposuretime)

    def get_gain(self):
        attr = self.controller.get_attribute("gain")
        return attr.value, attr.time.totime(), attr.quality

    def set_gain(self, new_gain):
        self.debug_stream("In set_gain: New value {0}".format(new_gain))
        self.controller.write_attribute("gain", new_gain)

    def get_width(self):
        attr = self.controller.get_attribute("width")
        return attr

    def get_peak(self):
        attr = self.controller.get_attribute("peak")
        return attr

    def get_satlvl(self):
        attr = self.controller.get_attribute("satlvl")
        return attr


if __name__ == "__main__":
    pt.server.server_run((FrogControllerDS, ))
