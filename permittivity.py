import math
import cmath
import numpy as np
import sympy as sp
from sympy import *
import matplotlib.pyplot as plt
from pandas import DataFrame

x, y, z = symbols('x y z')

epsilonvac = 8.85418782 * 10 ** (-12)
C1 =  np.linspace(0.001,0.03,100) #### volume concentration of graphene ###
C0 = 1 - C1  #### volume concentration of epoxy ######

####################INPUTS###########################
# Graphene#
#Density (g/c^3)#
pg = 2.2              
# aspect ratio
aspectr =  2.54e-4 #1.26e-3 #0.0014    #2.52e-4 # 7.9e-4 #0.00028   # 0.0014 for 5 micron, 0.00028 for 25 micron  # Azadeh 25 - 0.008613 & 5 - 0.004649 #Thermal Diffusivity Mapping of Graphene Based Polymer Nanocomposites 25 - 2.54*10^-4 5 - 1.26*10^-3
# thickness
Gthick = 8e-9

# parallel permitivity
epsilon1 = 31.5 * (epsilonvac)
# perpendicular permitivity
epsilon3 = 0.67 * (epsilon1)

# Polymer#
#Density cured (g/c^3)#
pp = 1.144
# permitivity
epsilon0 = 3.0666 * (epsilonvac)

# Aglomerate#
# aspect ratio
aspectrR = 0.1
# a
a = 0.8
# b
b = 0.05

# Interfacial (Initial)#
# thickness
h = 3 * 10 ** (-10)
# permitivity
epsilonint0 = 7 * (epsilonvac)
# permtivity at high freq
epsilonint_highfreq0 = 0.1*(epsilonvac)

# Nano-capacitance#
# scale paramter 1
gammaepsilon1 = 2.5*10**(-6)
# scale paramter 2 (at frequency aproaching infinity)a
gammaepsilon2 = 2.5*10**(-5)

# Electron hopping#
# debye relaxation time
trel = 2*10**(-3)

## a = (CR1*CR)/C1 ###### Selective localization ####
CR = (a / (b + C1 * (1 - b))) * C1  ### concentration of rich sections as dependent on graphene concentration####
CP = 1 - CR  # concentartion of poor section
CP1 = ((1 - a) * (b + C1 * (1 - b)) / (-C1 * a + (b + C1 * (
            1 - b)))) * C1  ### concentraion of graphene in the poor section as dependent on the total amount of graphene###
CR1 = b + C1 * (1 - b)  # concentration of graphene in the rich section as dependent on total amount of graphene
CP0 = 1 - CP1  # concentration of epoxy in the poor section
CR0 = 1 - CR1  # concentration of epoxy in the rich section

########## eschelby tensor of agglomerate ##########################
if (aspectrR < 1):
    S11R = S22R = ((aspectrR) / (2 * (1 - aspectrR ** 2) ** (3 / 2))) * (
                math.acos(aspectrR) - aspectrR * (1 - aspectrR ** 2) ** (1 / 2))
else:
    S11R = S22R = ((aspectrR) / (2 * (aspectrR ** 2 - 1) ** (3 / 2))) * (
                aspectrR * (aspectrR ** 2 - 1) ** (1 / 2) - math.acosh(aspectrR))

S33R = 1 - 2 * S11R

########## eschelby tensor of graphene ##########################
if (aspectr < 1):
    S11 = S22 = ((aspectr) / (2 * (1 - aspectr ** 2) ** (3 / 2))) * (
                math.acos(aspectr) - aspectr * (1 - aspectr ** 2) ** (1 / 2))
else:
    S11 = S22 = ((aspectr) / (2 * (aspectr ** 2 - 1) ** (3 / 2))) * (
                aspectr * (aspectr ** 2 - 1) ** (1 / 2) - math.acosh(aspectr))

S33 = 1 - 2 * S11

####percolation threshold with agglomerate modification####
percR = (9 * S33R * (1 - S33R)) / (-9 * S33R ** 2 + 15 * S33R + 2)
perc = (b * percR) / (a - (1 - b) * percR)

###########################IMPERFECT INTERFACIAL TERMS ##############
cint = (Gthick / 2) * ((Gthick / (2 * aspectr)) ** 2) / (((Gthick / 2) + h) * ((Gthick / (
            2 * aspectr)) + h) ** 2)  #####volume inclusion of the interlayer with respect to the whole coated inclusion####


###########ELECTRON TUNNELLING / NANOCAPICITANCE MODIFICATION TO INTERLAYER CONDUCTIVITTY AND PERMITTIVTY##########
#### Cauchy's cumulative probablistic function #####
def cauchy(c1, perc, gamma):
    return (1 / math.pi) * np.arctan((c1 - perc) / gamma) + (1 / 2)

F1epsilon = cauchy(1, perc, gammaepsilon1)
F2epsilon = cauchy(C1, perc, gammaepsilon1)
F3epsilon = cauchy(1, perc, gammaepsilon1)
F4epsilon = cauchy(0, perc, gammaepsilon1)
F1epsilon_highfreq = cauchy(1, perc, gammaepsilon2)
F2epsilon_highfreq = cauchy(C1, perc, gammaepsilon2)
F3epsilon_highfreq = cauchy(1, perc, gammaepsilon2)
F4epsilon_highfreq = cauchy(0, perc, gammaepsilon2)
####Common resistance like function####

tauepsilon = (F1epsilon - F2epsilon) / (F3epsilon - F4epsilon)
tauepsilon_highfreq = (F1epsilon_highfreq - F2epsilon_highfreq) / (F3epsilon_highfreq - F4epsilon_highfreq)
##################################

epsilonint_static = epsilonint0 / tauepsilon
epsilonint_highfreq_static = epsilonint_highfreq0 / tauepsilon_highfreq

def emt(C0,sigma0,sigma1,sigma3,C1,S11,S33):
 A = C0*(sigma0 - x)
 B = x + (1/3)*(sigma0 - x)
 C = (2/3)*C1*(sigma1-x)
 D = x + S11*(sigma1 - x)
 E = (1/3)*C1*(sigma3 - x)
 F = x + S33*(sigma3 - x)
 num =  poly(A*D*F + B*C*F + B*D*E,x)
 coeffs = num.coeffs()
 roots = np.roots(coeffs)
 for item in roots:
   if (complex(item).real > 0):
    roots = complex(item).real
 return roots


### Dyre's hopping and Debye's relaxtion modification to interlayer conductivity and permittivity###
FREQ_array = 6*10**(9)
omega = 2*math.pi*FREQ_array
epsilonint_freq = epsilonint_highfreq_static + ((epsilonint_static-epsilonint_highfreq_static)/(1+(omega**(2))*(trel**(2))))

############IMPERFECT INTERFACE MODIFICATION ############
def layered_graphene(sigmaint, sig):
   return (sigmaint * ((1) + ((1 - cint) * (sig - sigmaint)) / (cint * S11 * (sig - sigmaint) + sigmaint)))


##### permitivitty in both directions with isotropic interlayer coating ######
epsilon1coatf = layered_graphene(epsilonint_freq,epsilon1)  
epsilon3coatf = layered_graphene(epsilonint_freq,epsilon3) 

##### permitivitty in both directions with isotropic interlayer coating ######
epsilon1coat_static = layered_graphene(epsilonint_static, epsilon1)  
epsilon3coat_static = layered_graphene(epsilonint_static, epsilon3)  


####################################################################

epsilonR_array = np.empty(shape=(0))
epsilonP_array = np.empty(shape=(0))
epsilon_array = np.empty(shape=(0))

for cr0,cr1,epsilon1c,epsilon3c in zip(CR0,CR1,epsilon1coatf,epsilon3coatf):
 epsilonR = emt(cr0,epsilon0,epsilon1c,epsilon3c,cr1,S11,S33)
 epsilonR_array = np.append(epsilonR_array, [epsilonR], axis=0)


for cp0,cp1,epsilon1c,epsilon3c in zip(CP0,CP1,epsilon1coatf,epsilon3coatf):
 epsilonP = emt(cp0,epsilon0,epsilon1c,epsilon3c,cp1,S11,S33)
 epsilonP_array = np.append(epsilonP_array, [epsilonP], axis=0)


for cp,cr,epsilonr,epsilonp in zip(CP,CR,epsilonR_array,epsilonP_array):
 epsilon = emt(cp,epsilonp,epsilonr,epsilonr,cr,S11R,S33R)
 epsilon_array = np.append(epsilon_array, [epsilon], axis=0)

C1W = (pg*C1)/((pg*C1)+(pp*(1-C1)))
percW = (pg*perc)/((pg*perc)+(pp*(1-perc)))

#plt.figure(1)
#plt.xlabel('graphene weight % (freq = 6Ghz)')
#plt.ylabel('log10(effective relative permitivitty)')
#plt.plot(C1W*100,np.log10(epsilon_array/epsilonvac))
#plt.show()


table = {"GnP weight %": (C1W*100),"Relative Perimittivity": (epsilon_array/epsilonvac),"percolaton threshold": (percW*100),}
df = DataFrame(table, columns= ["GnP weight %","Relative Perimittivity","percolation threshold"])
ep_results = df.to_csv(r'C:\Users\61432\Desktop\Codes\graphene nanocomposite modelling\graphene nanocomposite modelling\ep_result.csv',index=None,header=True)

print(percW*100)

