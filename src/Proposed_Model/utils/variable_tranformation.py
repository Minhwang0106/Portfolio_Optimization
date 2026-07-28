import pandas as pd
import numpy as np

class Ros_Trans:
    @staticmethod
    def transform (values):
        return -np.log(1-values)
    @staticmethod
    def inverse (values):
        return 1-np.exp(-values)

class Ato_Trans:
    @staticmethod
    def transform (values):
        return np.log(values)
    @staticmethod
    def inverse (values):
        return np.exp(values)

class Ate_Trans:
    @staticmethod
    def transform (values):
        return np.log(values-1)
    @staticmethod
    def inverse (values):
        return np.exp(values)+1
    
    