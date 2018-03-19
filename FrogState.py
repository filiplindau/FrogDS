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
from TangoTwisted import TangoAttributeFactory


logger = logging.getLogger("FrogState")
logger.setLevel(logging.DEBUG)


class FrogStateDispatcher(object):
    def __init__(self, controller):
        self.controller = controller
        self.stop_flag = False
        self.statehandler_dict()
        self.statehandler_dict["unknown"] = FrogStateUnknown
        self.statehandler_dict["device_connect"] = FrogStateDeviceConnect
        self.statehandler_dict["attribute_setup"] = FrogStateSetupAttributes
        self.statehandler_dict["scan"] = FrogStateScan
        self.statehandler_dict["analyse"] = FrogStateAnalyse
        self.statehandler_dict["idle"] = FrogStateIdle
        self.statehandler_dict["fault"] = FrogStateFault
        self.current_state = "unknown"

    def statehandler_dispatcher(self):
        prev_state = self.get_state()
        while self.stop_flag is False:
            try:
                state_name = self.get_state()
                state = self.statehandler_dict[state_name](self.controller)
            except KeyError:
                self.statehandler_dict["unknown"]
            state.state_enter(prev_state)
            state.run()     # <- this should be run in a loop either in state or here in dispatcher
            new_state = state.state_exit()
            self.set_state(new_state)
            prev_state = state_name

    def get_state(self):
        return self.current_state

    def set_state(self, state_name):
        self.current_state = state_name

    def stop(self):
        self.stop_flag = True


class FrogState(object):
    def __init__(self, controller):
        self.name = ""
        self.controller = controller
        self.logger = logging.getLogger("FrogState.FrogState")
        self.logger.setLevel(logging.DEBUG)
        self.deferred_list = list()
        self.next_state = None

    def state_enter(self, prev_state=None):
        pass

    def state_exit(self):
        for d in self.deferred_list:
            try:
                d.cancel()
            except defer.CancelledError:
                pass
        return self.next_state

    def run(self):
        pass

    def check_requirements(self, result):
        """
        If next_state is None: stay on this state, else switch state
        :return:
        """
        self.next_state = None
        return result

    def state_error(self, err):
        pass

    def get_name(self):
        return self.name

    def get_state(self):
        return self.name

    def send_message(self, msg):
        pass


class FrogStateDeviceConnect(FrogState):
    """
    Connect to tango devices needed for the frog.
    The names of the devices are stored in the controller.device_names list.
    Devices:
    motor
    spectrometer
    Devices are stored as TangoAttributeFactories in controller.device_factory_dict

    """
    def __init__(self, controller):
        FrogState.__init__(self, controller)
        self.name = "device_init"
        self.controller.device_factory_dict = dict()
        self.deferred_list = list()
        self.logger.name = self.name

    def state_enter(self, prev_state):
        dl = list()
        for dev_name in self.controller.device_names:
            fact = TangoAttributeFactory(dev_name)
            dl.append(fact.startFactory())
            self.controller.device_factory_dict[dev_name] = fact
        def_list = defer.DeferredList(dl)
        self.deferred_list.append(def_list)
        def_list.addCallbacks(self.check_requirements, self.state_error)

    def state_error(self, err):
        pass

    def check_requirements(self, result):
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
    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateScan(FrogState):
    """
    Start a FROG scan using scan_params parameters dict stored in the controller.
    scan_params["start_pos"]: initial motor position
    scan_params["step_size"]: motor step size
    scan_params["end_pos"]: motor end position
    scan_params["average"]: number of averages in each position
    """
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
    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateIdle(FrogState):
    """
    Wait for time for a new scan or a command. Parameters stored in controller.idle_params
    idle_params["scan_interval"]: time in seconds between scans
    """
    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateFault(FrogState):
    """
    Handle fault condition.
    """
    def __init__(self, controller):
        FrogState.__init__(self, controller)


class FrogStateUnknown(FrogState):
    """
    Limbo state.
    Wait and try to connect to devices.
    """
    def __init__(self, controller):
        FrogState.__init__(self, controller)
