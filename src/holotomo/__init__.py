from pkg_resources import get_distribution, DistributionNotFound

# from holotomo.solver_holo import *
# from holotomo.solver_tomo import *
from holotomo import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
