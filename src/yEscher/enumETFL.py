from enum import Enum
import os

class ETFL(Enum):
    CEFL = os.path.join(os.path.dirname(__file__), 'yeastModel/input_model/yeast8_cEFL_2584_enz_128_bins__20240209_125642.json')