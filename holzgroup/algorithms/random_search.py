import torch
import numpy as np


def ars(alpha, iterations, v, b, n, p):
    """
    :param alpha: ﻿step-size α
    :param iterations: ﻿number of directions sampled per iteration N
    :param v: ﻿standard deviation of the exploration noise ν
    :param b: ﻿number of top-performing directions to use b (b < N is allowed only for V1-t and V2-t)
    :param n: ﻿some dimensiom
    :param p: ﻿some dimension
    :return:
    """
    m_0, mu_0, sigma_0, j = initialize(n, p)
    while ending_condition_satisfied():



def initialize(n, p):
    return np.zeros((p, n)), np.zeros(n), np.identity(n), 0


def ending_condition_satisfied():
    return 0 < 1

