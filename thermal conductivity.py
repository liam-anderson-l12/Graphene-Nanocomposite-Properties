import math
import cmath
import numpy as np
import sympy as sp
from pylab import *
from sympy import *
import matplotlib.pyplot as plt
from tabulate import tabulate

x, y, z = symbols('x y z')

C1 =  np.linspace(0.001,0.03,100) #### volume concentration of graphene ###
C0 = 1 - C1  #### volume concentration of epoxy ######

####################INPUTS###########################
# Graphene#
#Density (g/c^3)#
pg = 2.2
# aspect ratio
aspectr =  7.9e-4    # Azadeh 25 - 0.008613 # 5 - 0.004649   # 18 layers. kaptiza paper. 25mic - 2.52e-4 & 5mic - 7.9e-4 (assuming same thickness)
# thickness
Gthick = 6.3 * (10 ** (-9))
# parallel thermal conductivity
k1 = 80 #3000
# perpendicular thermal conductivity
k3 = 0.8 #6


# Polymer#
#Density cured (g/c^3)#
pp = 1.144
# conductivity
k0 = 0.2

# Interfacial #
# thickness
h = 12 * (10 ** (-9))
# conductivity
kintfm = 0.04
kintff = 0.24

# cuachy's scale parameter#
# scale parameter
gamma = 0.008



########## eschelby tensor of graphene ##########################
if (aspectr < 1):
    S11 = S22 = ((aspectr) / (2 * (1 - aspectr ** 2) ** (3 / 2))) * (
                math.acos(aspectr) - aspectr * (1 - aspectr ** 2) ** (1 / 2))
else:
    S11 = S22 = ((aspectr) / (2 * (aspectr ** 2 - 1) ** (3 / 2))) * (
                aspectr * (aspectr ** 2 - 1) ** (1 / 2) - math.acosh(aspectr))

S33 = 1 - 2 * S11

####percolation threshold with agglomerate modification####
perc = (9 * S33 * (1 - S33)) / (-9 * S33 ** 2 + 15 * S33 + 2)


#######################interlayer volume concentration ##############
cint = 1 - (Gthick / 2) * ((Gthick / (2 * aspectr)) ** 2) / (((Gthick / 2) + h) * ((Gthick / (
            2 * aspectr)) + h) ** 2)  #####volume inclusion of the interlayer with respect to the whole coated inclusion####


############ layered graphene ############
def layered_graphene(kint0, k):
   return (kint0 * ((1) + ((1 - cint) * (k - kint0)) / (cint * S11 * (k - kint0) + kint0)))


########### F-F & F-M contact ##########
#### Cauchy's cumulative probablistic function #####
def cauchy(c1, perc, gamma):
    return (1 / math.pi) * np.arctan((c1 - perc) / gamma) + (1 / 2)


def emt(C0,k0,k1,k3,C1,S11,S33):
 A = C0*(k0 - x)
 B = x + (1/3)*(k0 - x)
 C = (2/3)*C1*(k1-x)
 D = x + S11*(k1 - x)
 E = (1/3)*C1*(k3 - x)
 F = x + S33*(k3 - x)
 num =  poly(A*D*F + B*C*F + B*D*E,x)
 coeffs = num.coeffs()
 roots = np.roots(coeffs)
 for item in roots:
   if (complex(item).real > 0):
    roots = complex(item).real
 return roots

F1 = cauchy(1, perc, gamma)
F2 = cauchy(C1, perc, gamma)
F3 = cauchy(1, perc, gamma)
F4 = cauchy(0, perc, gamma)

tau = (F1 - F2) / (F3 - F4)

kint = kintfm*(1-tau)+ kintff*(tau)

k1c = layered_graphene(kint, k1)
k3c = layered_graphene(kint, k3)

ke_array = []
for c0,c1,k1c,k3c in zip(C0,C1,k1c,k3c):
 ke = emt(c0,k0,k1c,k3c,c1,S11,S33)
 ke_array = np.append(ke_array, [ke], axis=0)

C1W = (pg*C1)/((pg*C1)+(pp*(1-C1)))

#plt.figure(1)
#plt.xlabel('graphene weight %')
#plt.ylabel('effective thermal conductivity [W/m·K]')
#plt.plot(C1W*100,ke_array)
#plt.show()

table = [["GnP weight %", C1W*100],["thermal conducutivity", ke_array]]
print(tabulate(table))

print(perc)