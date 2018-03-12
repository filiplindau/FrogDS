# -*- coding:utf-8 -*-
"""
Created on Feb 27, 2018

@author: Filip Lindau
"""
import threading
import time
import logging
import Queue
from concurrent.futures import Future
from twisted.internet import reactor, defer
from twisted.internet.protocol import Protocol, ClientFactory, Factory
from twisted.python.failure import Failure
import PyTango as tango
import PyTango.futures as tangof


logger = logging.getLogger("FrogController")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

# f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def deferred_from_future(future):
    d = defer.Deferred()

    def callback(future):
        e = future.exception()
        if e:
            d.errback(e)
            return

        d.callback(future.result())

    future.add_done_callback(callback)
    return d


class TangoAttributeProtocol(Protocol):
    def __init__(self, operation, name, data=None):
        self.data = data
        self.result_data = None
        self.operation = operation
        self.name = name
        self.d = None
        self.factory = None

        self.logger = logging.getLogger("FrogController.Protocol_{0}_{1}".format(operation.upper(), name))
        self.logger.setLevel(logging.DEBUG)

    def makeConnection(self, transport=None):
        self.logger.debug("Protocol {0} make connection".format(self.name))
        if self.operation == "read":
            self.d = deferred_from_future(self.factory.device.read_attribute(self.name, wait=False))
        elif self.operation == "write":
            self.d = deferred_from_future(self.factory.device.write_attribute(self.name, self.data, wait=False))
        elif self.operation == "command":
            self.d = deferred_from_future(self.factory.device.command_inout(self.name, self.data, wait=False))
        self.d.addCallbacks(self.dataReceived, self.connectionLost)
        return self.d

    def dataReceived(self, data):
        self.result_data = data
        self.logger.debug("Received data {0}".format(data))
        return data

    def connectionLost(self, reason):
        self.logger.debug("Connection lost, reason {0}".format(reason))
        return reason


class TangoAttributeFactory(Factory):
    protocol = TangoAttributeProtocol

    def __init__(self, device_name):
        self.device_name = device_name
        self.device = None
        self.connected = False
        self.d = None
        self.proto_list = list()
        self.attribute_dict = dict()

        self.logger = logging.getLogger("FrogController.Factory_{0}".format(device_name))
        self.logger.setLevel(logging.CRITICAL)

    def startFactory(self):
        self.logger.info("Starting TangoAttributeFactory")
        self.d = deferred_from_future(tangof.DeviceProxy(self.device_name, wait=False))
        self.d.addCallbacks(self.connection_success, self.connection_fail)

    def buildProtocol(self, operation, name, data=None, d=None):
        """
        Create a TangoAttributeProtocol that sends a Tango operation to the factory deviceproxy.

        :param operation: Tango attribute operation, e.g. read, write, command
        :param name: Name of Tango attribute
        :param data: Data to send to Tango device, if any
        :param d: Optional deferred to add the result of the Tango operation to
        :return: Deferred that fires when the Tango operation is completed.
        """
        if self.connected is True:
            self.logger.info("Connected, create protocol and makeConnection")
            proto = self.protocol(operation, name, data)
            proto.factory = self
            self.proto_list.append(proto)
            df = proto.makeConnection()
            df.addCallbacks(self.data_received, self.protocol_fail)
            if d is not None:
                df.addCallback(d)
        else:
            self.logger.debug("Not connected yet, adding to connect callback")
            # df = defer.Deferred()
            self.d.addCallbacks(self.build_protocol_cb, self.connection_fail, callbackArgs=[operation, name, data])
            df = self.d

        return df

    def build_protocol_cb(self, result, operation, name, data, df=None):
        """
        We need this extra callback for buildProtocol since the first argument
        is always the callback result.

        :param result: Result from deferred callback. Ignore.
        :param operation: Tango attribute operation, e.g. read, write, command
        :param name: Name of Tango attribute
        :param data: Data to send to Tango device, if any
        :param df: Optional deferred to add the result of the Tango operation to
        :return: Deferred that fires when the Tango operation is completed.
        """
        self.logger.debug("Now call build protocol")
        d = self.buildProtocol(operation, name, data, df)
        return d

    def connection_success(self, result):
        self.logger.debug("Connected to deviceproxy")
        self.connected = True
        self.device = result

    def connection_fail(self, err):
        self.logger.error("Failed to connect to device. {0}".format(err))
        self.device = None
        self.connected = False
        fail = Failure(err)
        return err

    def protocol_fail(self, err):
        self.logger.error("Failed to do attribute operation on device {0}: {1}".format(self.device_name, err))
        fail = Failure(err)
        return fail

    def data_received(self, result):
        self.logger.debug("Data received: {0}".format(result))
        try:
            self.attribute_dict[result.name] = result
        except AttributeError:
            pass
        return result

    def get_attribute(self, name):
        if name in self.attribute_dict:
            d = defer.Deferred()
            d.callback(self.attribute_dict[name])
        else:
            self.logger.debug("Attribute not in dictionary, retrieve it from device")
            d = self.buildProtocol("read", name)
        return d


class TangoAttributeReader(object):
    def __init__(self, device_name):
        self.device_name = device_name

    def doRead(self):
        pass


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

        self.device_factories = dict()
        self.device_factories["spectrometer"] = TangoAttributeFactory(spectrometer_name)
        self.device_factories["motor"] = TangoAttributeFactory(motor_name)

        self.logger = logging.getLogger("FrogController.Controller")
        self.logger.setLevel(logging.DEBUG)

        for dev_fact in self.device_factories:
            self.device_factories[dev_fact].doStart()

    def read_attribute(self, name, device_name):
        self.logger.info("Read attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factories[device_name]
            d = factory.buildProtocol("read", name)
        else:
            self.logger.error("Device name {0} not found among {1}".format(device_name, self.device_factories))
            err = tango.DevError(reason="Device {0} not used".format(device_name),
                                 severety=tango.ErrSeverity.ERR,
                                 desc="The device is not in the list of devices used by this controller",
                                 origin="read_attribute")
            d = Failure(tango.DevFailed(err))
        return d

    def write_attribute(self, name, device_name, data):
        self.logger.info("Write attribute \"{0}\" on \"{1}\"".format(name, device_name))
        if device_name in self.device_names:
            factory = self.device_factories[device_name]
            d = factory.buildProtocol("write", name, data)
        else:
            self.logger.error("Device name {0} not found among {1}".format(device_name, self.device_factories))
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


class LoopingCall(object):
    def __init__(self, loop_callable, *args, **kw):
        self.f = loop_callable
        self.args = args
        self.kw = kw
        self.running = False
        self.interval = 0.0
        self.starttime = None
        self._deferred = None
        self._runAtStart = False
        self.call = None

    def start(self, interval, now=True):
        """
        Start running function every interval seconds.
        @param interval: The number of seconds between calls.  May be
        less than one.  Precision will depend on the underlying
        platform, the available hardware, and the load on the system.
        @param now: If True, run this call right now.  Otherwise, wait
        until the interval has elapsed before beginning.
        @return: A Deferred whose callback will be invoked with
        C{self} when C{self.stop} is called, or whose errback will be
        invoked when the function raises an exception or returned a
        deferred that has its errback invoked.
        """
        if self.running is True:
            self.stop()

        if interval < 0:
            raise ValueError("interval must be >= 0")
        self.running = True
        # Loop might fail to start and then self._deferred will be cleared.
        # This why the local C{deferred} variable is used.
        deferred = self._deferred = defer.Deferred()
        self.starttime = time.time()
        self.interval = interval
        self._runAtStart = now
        if now:
            self()
        else:
            self._schedule_from(self.starttime)
        return deferred

    def stop(self):
        """Stop running function.
        """
        assert self.running, ("Tried to stop a LoopingCall that was "
                              "not running.")
        self.running = False
        if self.call is not None:
            self.call.cancel()
            self.call = None
            d, self._deferred = self._deferred, None
            d.callback(self)

    def reset(self):
        """
        Skip the next iteration and reset the timer.
        @since: 11.1
        """
        assert self.running, ("Tried to reset a LoopingCall that was "
                              "not running.")
        if self.call is not None:
            self.call.cancel()
            self.call = None
            self.starttime = time.time()
            self._schedule_from(self.starttime)

    def __call__(self):
        def cb(result):
            if self.running:
                self._schedule_from(time.time())
            else:
                d, self._deferred = self._deferred, None
                d.callback(self)

        def eb(failure):
            self.running = False
            d, self._deferred = self._deferred, None
            d.errback(failure)

        self.call = None
        d = defer.maybeDeferred(self.f, *self.args, **self.kw)
        d.addCallback(cb)
        d.addErrback(eb)

    def _schedule_from(self, when):
        """
        Schedule the next iteration of this looping call.
        @param when: The present time from whence the call is scheduled.
        """

        def how_long():
            # How long should it take until the next invocation of our
            # callable?  Split out into a function because there are multiple
            # places we want to 'return' out of this.
            if self.interval == 0:
                # If the interval is 0, just go as fast as possible, always
                # return zero, call ourselves ASAP.
                return 0
            # Compute the time until the next interval; how long has this call
            # been running for?
            running_for = when - self.starttime
            # And based on that start time, when does the current interval end?
            until_next_interval = self.interval - (running_for % self.interval)
            # Now that we know how long it would be, we have to tell if the
            # number is effectively zero.  However, we can't just test against
            # zero.  If a number with a small exponent is added to a number
            # with a large exponent, it may be so small that the digits just
            # fall off the end, which means that adding the increment makes no
            # difference; it's time to tick over into the next interval.
            if when == when + until_next_interval:
                # If it's effectively zero, then we need to add another
                # interval.
                return self.interval
            # Finally, if everything else is normal, we just return the
            # computed delay.
            return until_next_interval

        self.call = threading.Timer(how_long(), self)
        self.call.start()


def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))


if __name__ == "__main__":
    fc = FrogController("sys/tg_test/1", "sys/tg_test/1")
    time.sleep(0)
    da = fc.read_attribute("double_scalar", "motor")
    da.addCallbacks(test_cb, test_err)
    da = fc.write_attribute("double_scalar_w", "motor", 10)
    da.addCallbacks(test_cb, test_err)

    da = fc.defer_later(3.0, fc.read_attribute, "short_scalar", "motor")
    da.addCallback(test_cb, test_err)

