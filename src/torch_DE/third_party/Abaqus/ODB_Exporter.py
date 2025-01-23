try:
    from odbAccess import *
    from abaqusConstants import *
    from symbolicConstants import *
    
except ImportError:
    raise ImportError('Could not import abaqus API packages. This likely means that this module is being called outside of Abaqus CAE. These commands must be run using Abaqus CAE Python')

from collections import namedtuple
import numpy as np
from typing import List,Tuple
from scipy.io import savemat,loadmat
from copy import deepcopy
from Abaqus.helper_functions import *
import pickle
