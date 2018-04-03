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
reload(fs)
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
        self.setup_attr_params["speed"] = ("motor", "speed", 1.0)
        self.setup_attr_params["acceleration"] = ("motor", "acceleration", 36.0)
        self.setup_attr_params["step_per_unit"] = ("motor", "step_per_unit", 324.0)

        self.idle_params = dict()
        self.idle_params["scan_interval"] = 10.0

        self.scan_params = dict()
        self.scan_params["start_pos"] = 8.3
        self.scan_params["step_size"] = 0.01
        self.scan_params["end_pos"] = 8.8
        self.scan_params["average"] = 1

        self.analyse_params = dict()
        self.analyse_params["method"] = "SHG"
        self.analyse_params["size"] = 128
        self.analyse_params["iterations"] = 30
        self.analyse_params["roi"] = "full"
        self.analyse_params["threshold"] = 0.01
        self.analyse_params["background_subtract"] = False

        self.logger = logging.getLogger("FrogController.Controller")
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("FrogController.__init__")

        self.state_lock = threading.Lock()
        self.status = ""
        self.state = "unknown"

        if start is True:
            self.device_factory_dict["spectrometer"] = TangoAttributeFactory(spectrometer_name)
            self.device_factory_dict["motor"] = TangoAttributeFactory(motor_name)

            for dev_fact in self.device_factory_dict:
                self.device_factory_dict[dev_fact].startFactory()

    def read_attribute(self, name, device_name):
        self.logger.info("Read attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factory_dict[device_name]
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
            factory = self.device_factory_dict[device_name]
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
        if dev_name in self.device_factory_dict:
            factory = self.device_factory_dict[dev_name]
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

    def get_status(self):
        with self.state_lock:
            st = self.status
        return st

    def set_status(self, status_msg):
        with self.state_lock:
            self.status = status_msg

    # def start_scan(self, scan_attr, start, stop, step, meas_attr):
    #     self.logger.info("Starting scan of {0} from {1} to {2} measuring {3}".format(scan_attr.upper(),
    #                                                                                  start, stop, meas_attr.upper()))
    #     scan_pos = start
    #     d0 = self.check_attribute(scan_attr, "motor", scan_pos, 0.1, 3.0, tolerance=scan_pos*1e-4, write=True)
    #     # d1 = self.read_attribute(meas_attr, "spectrometer")
    #     d0.addCallback(self.scan_arrive)
    #     d0.addCallback(lambda _: self.read_attribute(meas_attr, "spectrometer"))
    #     d0.addCallback(self.meas_scan)
    #
    #     return d0
    #
    # def scan_arrive(self, result):
    #     try:
    #         self.scan_pos = result.value
    #     except AttributeError:
    #         self.logger.error("Scan arrive not returning attribute: {0}".format(result))
    #         return False
    #
    #     self.logger.info("Scan arrive at pos {0}".format(self.scan_pos))
    #     return result
    #
    # def meas_scan(self, result):
    #     try:
    #         measure_value = result.value
    #     except AttributeError:
    #         self.logger.error("Measurement not returning attribute: {0}".format(result))
    #         return False
    #     self.logger.info("Measure at scan pos {0} result: {1}".format(self.scan_pos, measure_value))


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

