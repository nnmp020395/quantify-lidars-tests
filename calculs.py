""" 
Ce script est pour intégrer les formulaires physique de calculs sous forme des fonctions 
"""

import numpy as np 
import pandas as pd 
import math

class extinction_coef:

    ns = 1.00028571
    rho_n = 0.03
    #-------------
    Na = 6.02e23 #Avodrago
    R = 8.31451
    P = 101325 #Pa
    T = 273.15 + 15 #K
    numb_density = (P*Na)/(T*R)*1e-6
    Nso = 2.54743e19
    #-------------
    W = 355e-7

    math.pi