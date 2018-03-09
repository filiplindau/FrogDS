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
import PyTango as tango
import PyTango.futures as tangof


logger = logging.getLogger("FrogController")
while len(logger.handlers):
    logger.removeHandler(logger.handlers[0])

f = logging.Formatter("%(asctime)s - %(module)s.   %(funcName)s - %(levelname)s - %(message)s")
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

    def makeConnection(self, transport=None):
        logger.debug("Protocol {0} make connection".format(self.name))
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
        logger.debug("Received data {0}".format(data))
        return data

    def connectionLost(self, reason):
        logger.debug("Connection lost, reason {0}".format(reason))


class TangoAttributeFactory(Factory):
    protocol = TangoAttributeProtocol

    def __init__(self, device_name):
        self.device_name = device_name
        self.device = None
        self.connected = False
        self.d = None
        self.proto_list = list()

    def startFactory(self):
        logger.debug("Starting TangoAttributeFactory")
        self.d = deferred_from_future(tangof.DeviceProxy(self.device_name, wait=False))
        self.d.addCallbacks(self.connection_success, self.connection_fail)

    def buildProtocol(self, operation, name, data, d=None):
        """
        Create a TangoAttributeProtocol that sends a Tango operation to the factory deviceproxy.

        :param operation: Tango attribute operation, e.g. read, write, command
        :param name: Name of Tango attribute
        :param data: Data to send to Tango device, if any
        :param d: Optional deferred to add the result of the Tango operation to
        :return: Deferred that fires when the Tango operation is completed.
        """
        if self.connected is True:
            logger.debug("Connected, create protocol and makeConnection")
            proto = self.protocol(operation, name, data)
            proto.factory = self
            self.proto_list.append(proto)
            df = proto.makeConnection()
            df.addCallbacks(self.data_received, self.connection_fail)
            if d is not None:
                df.addCallback(d)
        else:
            logger.debug("Not connected yet, adding to connect callback")
            df = defer.Deferred()
            self.d.addCallback(self.build_protocol_cb, operation, name, data, df)

        return df

    def build_protocol_cb(self, result, operation, name, data, df):
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
        logger.debug("Now call build protocol")
        d = self.buildProtocol(operation, name, data, df)
        return d

    def connection_success(self, result):
        logger.debug("Connected to deviceproxy")
        self.connected = True
        self.device = result

    def connection_fail(self, err):
        logger.error("Failed to connect to device. {0}".format(err))
        self.device = None
        self.connected = False

    def data_received(self, result):
        logger.debug("Data received: {0}".format(result))
        return result


class TangoAttributeReader(object):
    def __init__(self, device_name):
        self.device_name = device_name

    def doRead(self):
        pass


class FrogController(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    taf = TangoAttributeFactory("sys/tg_test/1")
    taf.doStart()
    taf.buildProtocol("read", "state")
