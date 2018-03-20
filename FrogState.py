# -*- coding:utf-8 -*-
"""
Created on Feb 27, 2018

@author: Filip Lindau

All data is in the controller object. The state object only stores data needed to keep track
of state progress, such as waiting deferreds, and scan progress.
When a state transition is started, a new state object for that state is instantiated.
The state name to class table is stored in a dict.
"""

import threading
import time
import logging
import traceback
from twisted.internet import reactor, defer, error
import PyTango.futures as tangof
import TangoTwisted
reload(TangoTwisted)
from TangoTwisted import TangoAttributeFactory, defer_later


logger = logging.getLogger("FrogState")
logger.setLevel(logging.DEBUG)
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

# f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
f = logging.Formatter("%(asctime)s - %(name)s.   %(funcName)s - %(levelname)s - %(message)s")
fh = logging.StreamHandler()
fh.setFormatter(f)
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


class FrogStateDispatcher(object):
    def __init__(self, controller):
        self.controller = controller
        self.stop_flag = False
        self.statehandler_dict = dict()
        self.statehandler_dict[FrogStateUnknown.name] = FrogStateUnknown
        self.statehandler_dict[FrogStateDeviceConnect.name] = FrogStateDeviceConnect
        self.statehandler_dict[FrogStateSetupAttributes.name] = FrogStateSetupAttributes
        self.statehandler_dict[FrogStateScan.name] = FrogStateScan
        self.statehandler_dict[FrogStateAnalyse.name] = FrogStateAnalyse
        self.statehandler_dict[FrogStateIdle.name] = FrogStateIdle
        self.statehandler_dict[FrogStateFault] = FrogStateFault
        self.current_state = FrogStateUnknown.name
        self._state_obj = None
        self._state_thread = None

        self.logger = logging.getLogger("FrogState.FrogStateDispatcher")
        self.logger.setLevel(logging.DEBUG)

    def statehandler_dispatcher(self):
        self.logger.info("Entering state handler dispatcher")
        prev_state = self.get_state()
        while self.stop_flag is False:
            try:
                state_name = self.get_state()
                self.logger.debug("New state: {0}".format(state_name.upper()))
                self._state_obj = self.statehandler_dict[state_name](self.controller)
            except KeyError:
                self.statehandler_dict[FrogStateUnknown.name]
            self._state_obj.state_enter(prev_state)
            self._state_obj.run()     # <- this should be run in a loop either in state
            new_state = self._state_obj.state_exit()
            self.set_state(new_state)
            prev_state = state_name
        self._state_thread = None

    def get_state(self):
        return self.current_state

    def set_state(self, state_name):
        self.logger.info("Current state: {0}, set new state {1}".format(self.current_state.upper(), state_name.upper()))
        self.current_state = state_name

    def stop(self):
        self.logger.info("Stop state handler thread")
        self._state_obj.stop_run()
        self.stop_flag = True

    def start(self):
        self.logger.info("Start state handler thread")
        if self._state_thread is not None:
            self.stop()
        self._state_thread = threading.Thread(target=self.statehandler_dispatcher)
        self._state_thread.start()


class FrogState(object):
    name = ""

    def __init__(self, controller):
        self.controller = controller
        self.logger = logging.getLogger("FrogState.{0}".format(self.name.upper()))
        # self.logger.name =
        self.logger.setLevel(logging.DEBUG)
        self.deferred_list = list()
        self.next_state = None
        self.cond_obj = threading.Condition()

    def state_enter(self, prev_state=None):
        self.logger.info("Entering state {0}".format(self.name.upper()))

    def state_exit(self):
        self.logger.info("Exiting state {0}".format(self.name.upper()))
        for d in self.deferred_list:
            try:
                d.cancel()
            except defer.CancelledError:
                pass
        return self.next_state

    def run(self):
        self.logger.info("Entering run")
        with self.cond_obj:
            self.cond_obj.wait()
        self.logger.debug("Exiting run")

    def check_requirements(self, result):
        """
        If next_state is None: stay on this state, else switch state
        :return:
        """
        self.next_state = None
        return result

    def state_error(self, err):
        self.logger.error("Error {0} in state {1}".format(err, self.name.upper()))

    def get_name(self):
        return self.name

    def get_state(self):
        return self.name

    def send_message(self, msg):
        pass

    def stop_run(self):
        self.logger.info("Notify condition to stop run")
        with self.cond_obj:
            self.cond_obj.notify_all()


class FrogStateDeviceConnect(FrogState):
    """
    Connect to tango devices needed for the frog.
    The names of the devices are stored in the controller.device_names list.
    Devices:
    motor
    spectrometer
    Devices are stored as TangoAttributeFactories in controller.device_factory_dict

    """
    name = "device_connect"

    def __init__(self, controller):
        FrogState.__init__(self, controller)
        self.controller.device_factory_dict = dict()
        self.deferred_list = list()
        # self.logger = logging.getLogger("FrogState.FrogStateDeviceConnect")
        # self.logger.setLevel(logging.DEBUG)
        # self.logger.name = self.name

    def state_enter(self, prev_state):
        self.logger.info("Starting state {0}".format(self.name.upper()))
        dl = list()
        for key, dev_name in self.controller.device_names.items():
            self.logger.debug("Connect to device {0}".format(dev_name))
            fact = TangoAttributeFactory(dev_name)
            dl.append(fact.startFactory())
            self.controller.device_factory_dict[dev_name] = fact
        self.logger.debug("List of deferred device proxys: {0}".format(dl))
        def_list = defer.DeferredList(dl)
        self.deferred_list.append(def_list)
        def_list.addCallbacks(self.check_requirements, self.state_error)

    def check_requirements(self, result):
        self.logger.info("Check requirements result: {0}".format(result))
        self.next_state = "setup_attributes"
        self.stop_run()
        return "setup_attributes"


class FrogStateSetupAttributes(FrogState):
    """
    Setup attributes in the tango devices. Parameters stored in controller.setup_attr_params
    setup_attr_params["speed"]: motor speed
    setup_attr_params["acceleration"]: motor acceleration
    setup_attr_params["exposure"]: spectrometer exposure time
    setup_attr_params["trigger"]: spectrometer use external trigger (true/false)
    setup_attr_params["gain"]: spectrometer gain
    # setup_attr_params["roi"]: spectrometer camera roi (list [top, left, width, height])
    """
    name = "setup_attributes"

    def __init__(self, controller):
        FrogState.__init__(self, controller)
        self.controller.device_factory_dict = dict()
        self.deferred_list = list()
        self.logger.name = self.name


class FrogStateScan(FrogState):
    """
    Start a FROG scan using scan_params parameters dict stored in the controller.
    scan_params["start_pos"]: initial motor position
    scan_params["step_size"]: motor step size
    scan_params["end_pos"]: motor end position
    scan_params["average"]: number of averages in each position
    """
    name = "scan"

    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateAnalyse(FrogState):
    """
    Start FROG inversion of latest scan data. Parameters are stored in controller.analyse_params
    analyse_params["method"]: FROG method (SHG, TG, ...)
    analyse_params["size"]: size of FROG trace (128, 256, ...)
    analyse_params["iterations"]: number of iterations to calculate
    analyse_params["roi"]: region of interest of the
    analyse_params["threshold"]: threshold level for the normalized data
    analyse_params["background_subtract"]: do background subtraction using start pos spectrum
    """
    name = "analyse"

    def __init__(self, controller):
        FrogState.__init__(self, controller)



class FrogStateIdle(FrogState):
    """
    Wait for time for a new scan or a command. Parameters stored in controller.idle_params
    idle_params["scan_interval"]: time in seconds between scans
    """
    name = "idle"

    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateFault(FrogState):
    """
    Handle fault condition.
    """
    name = "fault"

    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateUnknown(FrogState):
    """
    Limbo state.
    Wait and try to connect to devices.
    """
    name = "unknown"

    def __init__(self, controller):
        FrogState.__init__(self, controller)
        self.deferred_list = list()
        self.start_time = None
        self.wait_time = 1.0

    def state_enter(self, prev_state):
        self.logger.info("Starting state {0}".format(self.name.upper()))
        self.start_time = time.time()
        df = defer_later(self.wait_time, self.check_requirements, [None])
        self.deferred_list.append(df)
        df.addCallback(test_cb)

    def check_requirements(self, result):
        self.logger.info("Check requirements result {0} for state {1}".format(result, self.name.upper()))
        self.next_state = "device_connect"
        self.stop_run()


def test_cb(result):
    logger.debug("Returned {0}".format(result))


def test_err(err):
    logger.error("ERROR Returned {0}".format(err))
