from bluesky.simulation import ScreenIO

class ScreenDummy(ScreenIO):
    """
    Dummy class for the screen. Inherits from ScreenIO to make sure all the
    necessary methods are there. This class is there to reimplement the echo
    method so that console messages are ignored.
    """
    def echo(self, text='', flags=0):
        pass