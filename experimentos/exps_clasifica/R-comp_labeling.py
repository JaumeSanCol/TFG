import os
import numpy as np
import matplotlib.pyplot as plt
from somJ.som import SoM
from somJ.functions import *
import somJ.config as config
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score