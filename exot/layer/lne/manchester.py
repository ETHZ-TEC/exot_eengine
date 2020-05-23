"""Manchester line coding"""

from exot.exceptions import LayerMisconfigured

from .generic import GenericLineCoding

class ManchesterLineCoding(GenericLineCoding):
    def __init__(self, *args, **kwargs):

        # TODO use carrier from channel
        # Two valid conventions are accepted:
        # 1. IEEE 802.3 (0 = falling edge); 2. G. E. Thomas (0 = rising edge).
        if "style" in kwargs:
            style = kwargs.pop("style")
            if style == "IEEE":
                signal = {0: [-1, 0], 1: [0, -1], "carrier": [1, 0]}
            elif style == "Thomas":
                signal = {0: [0, -1], 1: [-1, 0], "carrier": [1, 0]}
            else:
                raise LayerMisconfigured("Manchester style can be 'IEEE' or 'Thomas'")
        else:
            signal = {0: [-1, 0], 1: [0, -1], "carrier": [0, 1]}

        kwargs["signal"] = signal
        super().__init__(*args, **kwargs)
