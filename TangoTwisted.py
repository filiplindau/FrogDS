# -*- coding:utf-8 -*-
"""
Created on Mar 19, 2018

@author: Filip Lindau
"""
import threading
import time
import logging
import traceback
from twisted.internet import reactor, defer, error
from twisted.internet.protocol import Protocol, ClientFactory, Factory
from twisted.python.failure import Failure, reflect
import PyTango.futures as tangof
import FrogController
import numpy as np
from scipy.signal import medfilt2d
# import sys, os, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, os.path.join(parentdir, "Frog\src"))
import FrogCalculationSimpleGP as FrogCalculation


logger = logging.getLogger("TangoTwisted")
logger.setLevel(logging.DEBUG)
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
    def __init__(self, operation, name, data=None, **kw):
        self.data = data
        self.kw = kw
        self.result_data = None
        self.operation = operation
        self.name = name
        self.d = None
        self.factory = None

        self._check_target_value = None
        self._check_period = 0.3
        self._check_timeout = 1.0
        self._check_tolerance = None
        self._check_starttime = None
        self._check_lastreadtime = None

        self.logger = logging.getLogger("FrogController.Protocol_{0}_{1}".format(operation.upper(), name))
        self.logger.setLevel(logging.CRITICAL)

    def makeConnection(self, transport=None):
        self.logger.debug("Protocol {0} make connection".format(self.name))
        if self.operation == "read":
            self.d = deferred_from_future(self.factory.device.read_attribute(self.name, wait=False))
        elif self.operation == "write":
            self.d = deferred_from_future(self.factory.device.write_attribute(self.name, self.data, wait=False))
        elif self.operation == "command":
            self.d = deferred_from_future(self.factory.device.command_inout(self.name, self.data, wait=False))
        elif self.operation == "check":
            self.d = self.check_attribute()
        self.d.addCallbacks(self.dataReceived, self.connectionLost)
        return self.d

    def dataReceived(self, data):
        self.result_data = data
        self.logger.debug("Received data {0}".format(data))
        return data

    def connectionLost(self, reason):
        self.logger.debug("Connection lost, reason {0}".format(reason))
        return reason

    # def check_attribute(self, attr_name, dev_name, target_value, period=0.3, timeout=1.0, tolerance=None, write=True):
    def check_attribute(self):
        """
        Check an attribute to see if it reaches a target value. Returns a deferred for the result of the
        check.
        Upon calling the function the target is written to the attribute if the "write" parameter is True.
        Then reading the attribute is polled with the period "period" for a maximum time of "timeout".
        If the read value is within tolerance, the callback deferred is fired.
        If the read value is outside tolerance after "timeout" time, the errback is fired.
        The maximum time to check is "timeout"

        :param attr_name: Tango name of the attribute to check, e.g. "position"
        :param dev_name: Tango device name to use, e.g. "gunlaser/motors/zaber01"
        :param target_value: Attribute value to wait for
        :param period: Polling period when checking the value
        :param timeout: Time to wait for the attribute to reach target value
        :param tolerance: Absolute tolerance for the value to be accepted
        :param write: Set to True if the target value should be written initially
        :return: Deferred that will fire depending on the result of the check
        """
        self.logger.info("Entering check_attribute")
        self.d = defer.Deferred()

        try:
            write = self.kw["write"]
        except KeyError:
            write = True
        try:
            self._check_timeout = self.kw["timeout"]
        except KeyError:
            self.logger.debug("No timeout specified, using 1.0 s")
            self._check_timeout = 1.0
        try:
            self._check_tolerance = self.kw["tolerance"]
        except KeyError:
            self._check_tolerance = None
            self.logger.debug("No tolerance specified, using None")
        try:
            self._check_period = self.kw["period"]
        except KeyError:
            self._check_period = 0.3
            self.logger.debug("No period specified, using 0.3 s")
        try:
            self._check_target_value = self.kw["target_value"]
        except KeyError:
            self._check_target_value = None
            self.logger.error("No target value specified")
            self.d.errback("Target value required")
            return self.d

        self._check_starttime = time.time()
        self._check_lastreadtime = self._check_starttime

        if write is True:
            self.logger.debug("Issuing initial write")
            dw = deferred_from_future(self.factory.device.write_attribute(self.name,
                                                                          self._check_target_value,
                                                                          wait=False))
            # Add a callback that starts the reading after write completes
            dw.addCallbacks(self._check_do_read, self._check_fail)

        # Return the deferred that will fire with the result of the check
        return self.d

    def _check_w_done(self, result):
        self.logger.info("Write done, result {0}".format(result))
        return result

    def _check_fail(self, err):
        self.logger.error("Error, result {0}".format(err))
        self.d.errback(err)
        return err

    def _check_do_read(self, result=None):
        self.logger.info("Issuing read ")
        dr = deferred_from_future(self.factory.device.read_attribute(self.name, wait=False))
        dr.addCallbacks(self._check_read_done, self._check_fail)

    def _check_read_done(self, result):
        self.logger.info("Read done, result {0}".format(result))
        t0 = time.time()
        # First check if we timed out. Then fire the errback function and exit.
        if t0 - self._check_starttime > self._check_timeout:
            self.logger.warning("Timeout exceeded")
            self.d.errback("timeout")
            return
        # Now try extracting the read value (sometimes this is None).
        try:
            val = result.value
        except AttributeError:
            self.d.errback("Read result error {0}".format(result))
            return

        # Check if the read value is within tolerance. Then fire the callback and exit.
        done = False
        if self._check_tolerance is None:
            if val == self._check_target_value:
                done = True
        else:
            if abs(val - self._check_target_value) < self._check_tolerance:
                done = True
        if done is True:
            self.logger.debug("Result {0} with tolerance of target value {1}".format(val, self._check_target_value))
            self.d.callback(result)
            return

        # Finally calculate the wait time until next read.
        last_duration = t0 - self._check_lastreadtime
        if last_duration > self._check_period:
            delay = 0
        else:
            delay = self._check_period - last_duration
        self.logger.debug("Delay until next read: {0}".format(delay))
        defer_later(delay, self._check_do_read)
        self._check_lastreadtime = t0


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
        return self.d

    def buildProtocol(self, operation, name, data=None, d=None, **kw):
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
            self.logger.debug("args: {0}, {1}, {2}, kw: {3}".format(operation, name, data, kw))
            proto = self.protocol(operation, name, data, **kw)
            proto.factory = self
            self.proto_list.append(proto)
            df = proto.makeConnection()
            df.addCallbacks(self.data_received, self.protocol_fail)
            if d is not None:
                df.addCallback(d)
        else:
            self.logger.debug("Not connected yet, adding to connect callback")
            # df = defer.Deferred()
            self.d.addCallbacks(self.build_protocol_cb, self.connection_fail, callbackArgs=[operation, name, data],
                                callbackKeywords=kw)
            df = self.d

        return df

    def build_protocol_cb(self, result, operation, name, data, df=None, **kw):
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
        d = self.buildProtocol(operation, name, data, df, **kw)
        return d

    def connection_success(self, result):
        self.logger.debug("Connected to deviceproxy")
        self.connected = True
        self.device = result

    def connection_fail(self, err):
        self.logger.error("Failed to connect to device. {0}".format(err))
        self.device = None
        self.connected = False
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
            self.logger.debug("Attribute {0} already read. Retrieve from dictionary.".format(name))
            d = defer.Deferred()
            d.callback(self.attribute_dict[name])
        else:
            self.logger.debug("Attribute {0} not in dictionary, retrieve it from device".format(name))
            d = self.buildProtocol("read", name)
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

        self.loop_deferred = defer.Deferred()

        self.logger = logging.getLogger("FrogController.LoopingCall")
        self.logger.setLevel(logging.DEBUG)

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

        self.logger.debug("Starting looping call")
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
                new_loop_deferred = defer.Deferred()
                for callback in self.loop_deferred.callbacks:
                    new_loop_deferred.callbacks.append(callback)
                self.loop_deferred.callback(result)
                self.loop_deferred = new_loop_deferred
            else:
                df, self._deferred = self._deferred, None
                df.callback(self)

        def eb(failure):
            self.running = False
            df, self._deferred = self._deferred, None
            df.errback(failure)

        self.call = None
        self.logger.debug("Calling function")
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


class DeferredCondition(object):
    def __init__(self, condition, cond_callable, *args, **kw):
        if "result" in condition:
            self.condition = condition
        else:
            self.condition = "result " + condition
        self.cond_callable = cond_callable
        self.args = args
        self.kw = kw
        self.logger = logging.getLogger("FrogController.DeferredCondition")
        self.logger.setLevel(logging.WARNING)

        self.running = False
        self.call_timer = None
        self._deferred = None
        self.starttime = None
        self.interval = None
        self.clock = None

    def start(self, interval, timeout=None):
        if self.running is True:
            self.stop()

        self.logger.debug("Starting checking condition {0}".format(self.condition))
        if interval < 0:
            raise ValueError("interval must be >= 0")
        self.running = True
        # Loop might fail to start and then self._deferred will be cleared.
        # This why the local C{deferred} variable is used.
        deferred = self._deferred = defer.Deferred()
        if timeout is not None:
            self.clock = ClockReactorless()
            deferred.addTimeout(timeout, self.clock)
            deferred.addErrback(self.cond_error)
        self.starttime = time.time()
        self.interval = interval
        self._run_callable()
        return deferred

    def stop(self):
        """Stop running function.
        """
        assert self.running, ("Tried to stop a LoopingCall that was "
                              "not running.")
        self.running = False
        if self.call_timer is not None:
            self.call_timer.cancel()
            self.call_timer = None
            d, self._deferred = self._deferred, None
            d.callback(None)

    def _run_callable(self):
        self.logger.debug("Calling {0}".format(self.cond_callable))
        d = defer.maybeDeferred(self.cond_callable, *self.args, **self.kw)
        d.addCallbacks(self.check_condition, self.cond_error)

    def _schedule_from(self, when):
        """
        Schedule the next iteration of this looping call.
        @param when: The present time from whence the call is scheduled.
        """
        t = 0
        if self.interval > 0:
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
                t = self.interval
            # Finally, if everything else is normal, we just return the
            # computed delay.
            else:
                t = until_next_interval
        self.logger.debug("Scheduling new function call in {0} s".format(t))
        self.call = threading.Timer(t, self._run_callable)
        self.call.start()

    def check_condition(self, result):
        self.logger.debug("Checking condition {0} with result {1}".format(self.condition, result))
        if self.running is True:
            cond = eval(self.condition)
            self.logger.debug("Condition evaluated {0}".format(cond))
            if cond is True:
                d, self._deferred = self._deferred, None
                d.callback(result)
            else:
                self._schedule_from(time.time())
            return result
        else:
            return False

    def cond_error(self, err):
        self.logger.error("Condition function returned error {0}".format(err))
        self.running = False
        if self.call_timer is not None:
            self.call_timer.cancel()
            self.call_timer = None
        d, self._deferred = self._deferred, None
        d.errback(err)
        return err


def defer_later(delay, delayed_callable, *a, **kw):
    # logger.info("Calling {0} in {1} seconds".format(delayed_callable, delay))

    def defer_later_cancel(deferred):
        delayed_call.cancel()

    d = defer.Deferred(defer_later_cancel)
    d.addCallback(lambda ignored: delayed_callable(*a, **kw))
    delayed_call = threading.Timer(delay, d.callback, [None])
    delayed_call.start()
    return d


class DelayedCallReactorless(object):
    # enable .debug to record creator call stack, and it will be logged if
    # an exception occurs while the function is being run
    debug = False
    _str = None

    def __init__(self, time, func, args, kw={}, cancel=None, reset=None,
                 seconds=time.time):
        """
        @param time: Seconds from the epoch at which to call C{func}.
        @param func: The callable to call.
        @param args: The positional arguments to pass to the callable.
        @param kw: The keyword arguments to pass to the callable.
        @param cancel: A callable which will be called with this
            DelayedCall before cancellation.
        @param reset: A callable which will be called with this
            DelayedCall after changing this DelayedCall's scheduled
            execution time. The callable should adjust any necessary
            scheduling details to ensure this DelayedCall is invoked
            at the new appropriate time.
        @param seconds: If provided, a no-argument callable which will be
            used to determine the current time any time that information is
            needed.
        """
        self.time, self.func, self.args, self.kw = time, func, args, kw
        self.resetter = reset
        self.canceller = cancel
        self.seconds = seconds
        self.cancelled = self.called = 0
        self.delayed_time = 0
        self.timer = None
        if self.debug:
            self.creator = traceback.format_stack()[:-2]
        self._schedule_call()

    def getTime(self):
        """Return the time at which this call will fire
        @rtype: C{float}
        @return: The number of seconds after the epoch at which this call is
        scheduled to be made.
        """
        return self.time + self.delayed_time

    def _schedule_call(self):
        if self.timer is not None:
            if self.timer.is_alive() is True:
                self.timer.cancel()
        self.timer = threading.Timer(self.time + self.delayed_time - time.time(), self._fire_call)
        self.timer.start()

    def _fire_call(self):
        self.called = True
        self.func(*self.args, **self.kw)

    def cancel(self):
        """Unschedule this call
        @raise AlreadyCancelled: Raised if this call has already been
        unscheduled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            if self.timer is not None:
                if self.timer.is_alive() is True:
                    self.timer.cancel()
            if self.canceller is not None:
                self.canceller(self)
            self.cancelled = 1
            if self.debug:
                self._str = str(self)
            del self.func, self.args, self.kw

    def reset(self, secondsFromNow):
        """Reschedule this call for a different time
        @type secondsFromNow: C{float}
        @param secondsFromNow: The number of seconds from the time of the
        C{reset} call at which this call will be scheduled.
        @raise AlreadyCancelled: Raised if this call has been cancelled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            new_time = self.seconds() + secondsFromNow
            if new_time < self.time:
                self.delayed_time = 0
                self.time = new_time
                if self.resetter is not None:
                    self.resetter(self)
            else:
                self.delayed_time = new_time - self.time
            self._schedule_call()

    def delay(self, secondsLater):
        """Reschedule this call for a later time
        @type secondsLater: C{float}
        @param secondsLater: The number of seconds after the originally
        scheduled time for which to reschedule this call.
        @raise AlreadyCancelled: Raised if this call has been cancelled.
        @raise AlreadyCalled: Raised if this call has already been made.
        """
        if self.cancelled:
            raise error.AlreadyCancelled
        elif self.called:
            raise error.AlreadyCalled
        else:
            self.delayed_time += secondsLater
            if self.delayed_time < 0:
                self.activate_delay()
                self.resetter(self)

    def activate_delay(self):
        self.time += self.delayed_time
        self.delayed_time = 0
        self._schedule_call()

    def active(self):
        """Determine whether this call is still pending
        @rtype: C{bool}
        @return: True if this call has not yet been made or cancelled,
        False otherwise.
        """
        return not (self.cancelled or self.called)

    def __le__(self, other):
        """
        Implement C{<=} operator between two L{DelayedCall} instances.
        Comparison is based on the C{time} attribute (unadjusted by the
        delayed time).
        """
        return self.time <= other.time

    def __lt__(self, other):
        """
        Implement C{<} operator between two L{DelayedCall} instances.
        Comparison is based on the C{time} attribute (unadjusted by the
        delayed time).
        """
        return self.time < other.time

    def __str__(self):
        if self._str is not None:
            return self._str
        if hasattr(self, 'func'):
            # This code should be replaced by a utility function in reflect;
            # see ticket #6066:
            if hasattr(self.func, '__qualname__'):
                func = self.func.__qualname__
            elif hasattr(self.func, '__name__'):
                func = self.func.func_name
                if hasattr(self.func, 'im_class'):
                    func = self.func.im_class.__name__ + '.' + func
            else:
                func = reflect.safe_repr(self.func)
        else:
            func = None

        now = self.seconds()
        L = ["<DelayedCall 0x%x [%ss] called=%s cancelled=%s" % (
                id(self), self.time - now, self.called,
                self.cancelled)]
        if func is not None:
            L.extend((" ", func, "("))
            if self.args:
                L.append(", ".join([reflect.safe_repr(e) for e in self.args]))
                if self.kw:
                    L.append(", ")
            if self.kw:
                L.append(", ".join(['%s=%s' % (k, reflect.safe_repr(v)) for (k, v) in self.kw.items()]))
            L.append(")")

        if self.debug:
            L.append("\n\ntraceback at creation: \n\n%s" % ('    '.join(self.creator)))
        L.append('>')

        return "".join(L)


class ClockReactorless(object):
    """
    Provide a deterministic, easily-controlled implementation of
    L{IReactorTime.callLater}.  This is commonly useful for writing
    deterministic unit tests for code which schedules events using this API.
    """

    rightNow = 0.0

    def __init__(self):
        self.calls = []
        self.timer = None

    def seconds(self):
        """
        Pretend to be time.time().  This is used internally when an operation
        such as L{IDelayedCall.reset} needs to determine a time value
        relative to the current time.
        @rtype: C{float}
        @return: The time which should be considered the current time.
        """
        self.rightNow = time.time()
        return self.rightNow

    def _sortCalls(self):
        """
        Sort the pending calls according to the time they are scheduled.
        """
        self.calls.sort(key=lambda a: a.getTime())

    def callLater(self, when, what, *a, **kw):
        """
        See L{twisted.internet.interfaces.IReactorTime.callLater}.
        """

        # def defer_later_cancel(deferred):
        #     delayed_call.cancel()
        #
        # dc = defer.Deferred(defer_later_cancel)
        # dc.addCallback(lambda ignored: what(*a, **kw))
        # delayed_call = threading.Timer(when, dc.callback, [None])
        # delayed_call.start()
        # self.calls.append(dc)
        # self._sortCalls()

        dc = DelayedCallReactorless(self.seconds() + when,
                                    what, a, kw,
                                    self.calls.remove,
                                    lambda c: None,
                                    self.seconds)
        self.calls.append(dc)
        self._sortCalls()
        return dc

    def getDelayedCalls(self):
        """
        See L{twisted.internet.interfaces.IReactorTime.getDelayedCalls}
        """
        return self.calls

    def advance(self, amount):
        """
        Move time on this clock forward by the given amount and run whatever
        pending calls should be run.
        @type amount: C{float}
        @param amount: The number of seconds which to advance this clock's
        time.
        """
        self.rightNow += amount
        # self._sortCalls()
        while self.calls and self.calls[0].getTime() <= self.seconds():
            call = self.calls.pop(0)
            call.called = 1
            call.func(*call.args, **call.kw)
            # self._sortCalls()

    def pump(self, timings):
        """
        Advance incrementally by the given set of times.
        @type timings: iterable of C{float}
        """
        for amount in timings:
            self.advance(amount)


def test_cb2(result):
    logger.info("Test CB2 result: {0}".format(result))
    return result


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

        self.logger = logging.getLogger("FrogController.Scan_{0}_{1}".format(self.scan_attr, self.meas_attr))
        self.logger.setLevel(logging.DEBUG)

        self.d = defer.Deferred()

        # Add errback handling!

    def start_scan(self):
        self.logger.info("Starting scan of {0} from {1} to {2} measuring {3}".format(self.scan_attr.upper(),
                                                                                     self.start_pos,
                                                                                     self.stop_pos,
                                                                                     self.meas_attr.upper()))
        scan_pos = self.start_pos
        tol = self.step * 0.1
        d0 = self.controller.check_attribute(self.scan_attr, self.scan_dev, scan_pos,
                                             0.1, 10.0, tolerance=tol, write=True)
        d0.addCallbacks(self.scan_arrive, self.scan_error_cb)
        # d0.addCallback(lambda _: self.controller.read_attribute(self.meas_attr, self.meas_dev))
        # d0.addCallback(self.meas_scan)

        return self.d

    def scan_step(self):
        self.logger.info("Scan step")
        tol = self.step * 0.1
        scan_pos = self.current_pos + self.step
        if scan_pos > self.stop_pos:
            self.logger.debug("Scan done!")
            self.d.callback([self.pos_data, self.meas_data])
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
        d0 = self.controller.read_attribute(self.meas_attr, self.meas_dev)
        # d0.addCallback(test_cb2)
        d0.addCallbacks(self.meas_scan, self.scan_error_cb)
        return True

    def meas_scan(self, result):
        self.logger.info("Meas scan result: {0}".format(result.value))
        measure_value = result.value
        # try:
        #     measure_value = result.value
        # except AttributeError:
        #     self.logger.error("Measurement not returning attribute: {0}".format(result))
        #     return False
        self.logger.info("Measure at scan pos {0} result: {1}".format(self.current_pos, measure_value))
        self.meas_data.append(measure_value)
        self.pos_data.append(self.current_pos)
        self.scan_step()
        return True

    def scan_error_cb(self, err):
        self.logger.error("Scan error: {0}".format(err))
        # Here we can handle the error if it is manageable or
        # propagate the error to the calling callback chain:
        if err.type in []:
            pass
        else:
            self.d.errback(err)


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
        self.time_data = None
        self.wavelength_data = None

        self.logger = logging.getLogger("FrogController.Analysis")
        self.logger.setLevel(logging.DEBUG)

        self.d = defer.Deferred()

    def start_analysis(self):
        self.logger.info("Starting up frog analysis")
        self.load_data()
        self.preprocess()
        self.convert_data()
        return self.d

    def load_data(self):
        self.logger.info("Loading frog data from scan")
        scan_result = self.controller.scan_result
        pos = np.array(scan_result[0])  # Vector containing the motor positions during the scan
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
        self.scan_raw_data = np.array(scan_result[1])
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
        np.clip(proc_data - thr, 0, None)     # In-place thresholding
        self.logger.debug("Scan data thresholded to {0}".format(thr))
        # Filtering
        kernel = int(self.controller.analyse_params["median_kernel"])
        if kernel > 1:
            proc_data = medfilt2d(proc_data, kernel)
        self.logger.debug("Scan data median filtered with kernel size {0}".format(kernel))
        self.scan_proc_data = proc_data

    def convert_data(self):
        """
        Create the reference intensity frog trace for the algorithm.
        We need the wavelength range and the time range.
        Also the size of the transform is required.
        :return:
        """
        self.logger.info("Converting scan data for FROG algorithm")
        tau_mean = (self.time_data[-1] + self.time_data[0]) / 2
        tau_start = self.time_data[0] - tau_mean
        tau_stop = self.time_data[-1] - tau_mean
        l_start = self.wavelength_data[0]
        l_stop = self.wavelength_data[-1]
        l0 = (l_stop + l_start) / 2
        n = self.controller.analyse_params["size"]
        dt_t = np.abs(tau_stop - tau_start) / n
        dt_l = np.abs(1.0 / (1.0 / l_start - 1.0 / l_stop) / 299792458.0)
        self.logger.debug("dt_t: {0}, dt_l: {1}".format(dt_t, dt_l))
        dt_p = self.controller.analyse_params["dt"]
        if dt_p is None:
            dt = dt_t
        else:
            dt = dt_p
        self.controller.frog_calc.init_pulsefield_random(n, dt, l0)
        self.logger.debug("Pulse field initialized with n={0}, dt={1} fs, l0={2} nm".format(n, dt*1e15, l0*1e9))
        self.controller.frog_calc.condition_frog_trace2(self.scan_proc_data, l_start, l_stop,
                                                        tau_start, tau_stop, n, thr=0)
        self.logger.debug("Frog trace conditioned to t-f space")

    def invert_frog_trace(self):
        pass


