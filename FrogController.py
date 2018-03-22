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
from TangoTwisted import TangoAttributeFactory, TangoAttributeProtocol, LoopingCall, DeferredCondition, ClockReactorless
import FrogState as fs
reload(fs)


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
    def __init__(self, spectrometer_name, motor_name):
        """
        Controller for running a scanning Frog device. Communicates with a spectrometer and a motor.


        :param spectrometer_name:
        :param motor_name:
        """
        self.device_names = dict()
        self.device_names["spectrometer"] = spectrometer_name
        self.device_names["motor"] = motor_name

        self.device_factory_dict = dict()
        # self.device_factory_dict["spectrometer"] = TangoAttributeFactory(spectrometer_name)
        # self.device_factory_dict["motor"] = TangoAttributeFactory(motor_name)

        self.logger = logging.getLogger("FrogController.Controller")
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("FrogController.__init__")

        self.state_lock = threading.Lock()
        self.status = ""
        self.state = "unknown"

        # for dev_fact in self.device_factory_dict:
        #     self.device_factory_dict[dev_fact].doStart()

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

    def check_attribute(self, attr_name, dev_name, target_value, period=0.3, retries=5, tolerance=None, write=True):
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
        :param retries: Number of retries before giving up
        :param tolerance: Absolute tolerance for the value to be accepted
        :param write: Set to True if the target value should be written initially
        :return: Deferred that will fire depending on the result of the check
        """
        if write is True:
            dw =

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


def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))


def test_timeout(result):
    logger.warning("TIMEOUT returned {0}".format(result))


if __name__ == "__main__":
    fc = FrogController("sys/tg_test/1", "sys/tg_test/1")
    time.sleep(0)
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

