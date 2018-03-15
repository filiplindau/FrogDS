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
        self.logger.setLevel(logging.WARNING)

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
                d, self._deferred = self._deferred, None
                d.callback(self)

        def eb(failure):
            self.running = False
            d, self._deferred = self._deferred, None
            d.errback(failure)

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


class DelayedCallReactorless(object):
    # enable .debug to record creator call stack, and it will be logged if
    # an exception occurs while the function is being run
    debug = False
    _str = None

    def __init__(self, time, func, args, kw, cancel, reset,
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
            newTime = self.seconds() + secondsFromNow
            if newTime < self.time:
                self.delayed_time = 0
                self.time = newTime
                self.resetter(self)
            else:
                self.delayed_time = newTime - self.time
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


def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))


def test_timeout(result):
    logger.warning("TIMEOUT returned {0}".format(result))


if __name__ == "__main__":
    fc = FrogController("sys/tg_test/1", "sys/tg_test/1")
    time.sleep(0)
    da = fc.read_attribute("double_scalar", "motor")
    da.addCallbacks(test_cb, test_err)
    da = fc.write_attribute("double_scalar_w", "motor", 10)
    da.addCallbacks(test_cb, test_err)

    # da = fc.defer_later(3.0, fc.read_attribute, "short_scalar", "motor")
    # da.addCallback(test_cb, test_err)

    # lc = LoopingCall(fc.read_attribute, "double_scalar_w", "motor")
    # dlc = lc.start(1)
    # dlc.addCallbacks(test_cb, test_err)
    # lc.loop_deferred.addCallback(test_cb)

    clock = ClockReactorless()

    defcond = DeferredCondition("result.value>15", fc.read_attribute, "double_scalar_w", "motor")
    dcd = defcond.start(1.0, timeout=3.0)
    dcd.addCallbacks(test_cb, test_err)

