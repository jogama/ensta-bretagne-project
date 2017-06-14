#!/usr/bin/python3

from scipy.stats import multivariate_normal as mnorm
import matplotlib.pyplot as plt
import numpy as np

'''
Author: Jonathan Garcia-Mallen
Creates two islands and has a uuv go around it.
The uuv takes measurements. They could be compass, odometry, or something else.
'''
