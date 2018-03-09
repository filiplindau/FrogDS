# FrogDS

Tango device server for measuring ultrafast laser pulses with the FROG technique. It uses Tango connected 
devices for spectrometer and motor.

All asynchronous communication is done with twisted.

The computation is done primarily with numpy, but can optionally be done with opencl. The algorithm can be 
selected as "vanilla" or general projections.

### Prerequisites

PyTango, twisted, numpy.


### Example usage

