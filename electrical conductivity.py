import math
import cmath
import numpy as np
import sympy as sp
from sympy import *
import matplotlib.pyplot as plt
from pandas import DataFrame

x, y, z = symbols('x y z')

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
# parallel conductivity
sig1 = 1 * 10 ** (7)      # from "Effect of Graphite Nanoplate Morphology on the Dispersion and Physical Properties of Polycarbonate" 25 mic - 22*10**(2) & 5 mic - 34*10**(2)

# perpendicular conductivity
sig3 = 1 * 10 **(2)      ## sig3 = l*sig1 with l = 10**(-3) from "An x-band theory of...... & similar" 

# Polymer#
#Density cured (g/c^3)#
pp = 1.144
# conductivity
sig0 = 1e-14

# Aglomerate#
# Thickness
Athick = 50 * 10**(-10)
# aspect ratio
aspectrR = 0.1
# a
a = 0.95  #0.8 for 5 micron #0.95 for 25micron
# b
b = 0.15    #0.05 for 5 micron #0.15 for 25micron

# Interfacial (Initial)#
# thickness
h = 3 * 10 ** (-10)
# conductivity
sigmaint0 = 6*10**(-7)

# Electron tunelling#
# scale parameter
gammasigma = 0.0002

# Electron hopping#
# characterisitc time
FREQ_array = 6e9
thop = 1.1*10**(-8)

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

#######Perc############
perc1 = (9 * S33 * (1 - S33)) / (-9 * S33 ** 2 + 15 * S33 + 2)


####percolation threshold with agglomerate modification####
percR = (9 * S33R * (1 - S33R)) / (-9 * S33R ** 2 + 15 * S33R + 2)
perc2 = (b * percR) / (a - (1 - b) * percR)

#####volume inclusion of the interlayer with respect to the whole coated inclusion####
cint = 1 -  (Gthick / 2) * ((Gthick / (2 * aspectr)) ** 2) / (((Gthick / 2) + h) * ((Gthick / (
            2 * aspectr)) + h) ** 2)  

#cint = 1 -  (Athick / 2) * ((Athick / (2 * aspectr)) ** 2) / (((Athick / 2) + h) * ((Athick / (
#            2 * aspectr)) + h) ** 2)  

###########ELECTRON TUNNELLING MODIFICATION TO INTERLAYER CONDUCTIVITTY ##########
#### Cauchy's cumulative probablistic function #####
def cauchy(c1, perc, gamma):
    return (1 / math.pi) * np.arctan((c1 - perc) / gamma) + (1 / 2)

F1sigma = cauchy(1, perc1, gammasigma)
F2sigma = cauchy(C1, perc1, gammasigma)
F3sigma = cauchy(1, perc1, gammasigma)
F4sigma = cauchy(0, perc1, gammasigma)

####Common resistance like function####
tausigma = (F1sigma - F2sigma) / (F3sigma - F4sigma)

##################################
sigmaint_static = sigmaint0 / tausigma

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


### Dyre's hopping modification to interlayer conductivity ###
omega = 2*math.pi*FREQ_array
m = omega*thop
Pomega = (omega*thop*(np.arctan(m))**(2))/((0.5*np.log(1+(m)**(2)))**(2)+(np.arctan(m))**(2))
sigmaint_freq = sigmaint_static*Pomega

############IMPERFECT INTERFACE MODIFICATION ############
def layered_graphene(sigmaint, sig):
   return (sigmaint * ((1) + ((1 - cint) * (sig - sigmaint)) / (cint * S11 * (sig - sigmaint) + sigmaint)))

##### conductivity in both directions with isotropic interlayer coating ######
sig1coatf = layered_graphene(sigmaint_freq,sig1)  
sig3coatf = layered_graphene(sigmaint_freq,sig3)  

##### conductivity in both directions with isotropic interlayer coating ###### (without electron hopping?)
sig1coat_static = layered_graphene(sigmaint_static, sig1)  
sig3coat_static = layered_graphene(sigmaint_static, sig3)  

###################### Agglomeration ##############################################

sigmaR_array = np.empty(shape=(0))
sigmaP_array = np.empty(shape=(0))
sigma_array = np.empty(shape=(0))

for cr0,cr1,sig1c,sig3c in zip(CR0,CR1,sig1coatf,sig3coatf):
 sigmaR = emt(cr0,sig0,sig1c,sig3c,cr1,S11,S33)
 sigmaR_array = np.append(sigmaR_array, [sigmaR], axis=0)

for cp0,cp1,sig1c,sig3c in zip(CP0,CP1,sig1coatf,sig3coatf):
 sigmaP = emt(cp0,sig0,sig1c,sig3c,cp1,S11,S33)
 sigmaP_array = np.append(sigmaP_array, [sigmaP], axis=0)

for cp,cr,sigr,sigp in zip(CP,CR,sigmaR_array,sigmaP_array):
 sigma = emt(cp,sigp,sigr,sigr,cr,S11R,S33R)
 sigma_array = np.append(sigma_array, [sigma], axis=0)

############## No Agglomeration ###################

#sigma_array = np.empty(shape=(0))

#for c0,c1,sig1c,sig3c in zip(C0,C1,sig1coat_static,sig3coat_static):
# sigmaR = emt(c0,sig0,sig1c,sig3c,c1,S11,S33)
# sigma_array = np.append(sigma_array, [sigmaR], axis=0)

C1W = (pg*C1)/((pg*C1)+(pp*(1-C1)))
percW = (pg*perc2)/((pg*perc2)+(pp*(1-perc2)))

#plt.figure(1)
#plt.xlabel('graphene weight % (freq = 0Ghz)')
#plt.ylabel('log10(effective conductivity)')
#plt.plot(C1W*100,np.log10(sigma_array))
#plt.show()



table = {"GnP weight %": (C1W*100),"electrical conductivity": (sigma_array),"percolaton threshold": (percW*100),}
df = DataFrame(table, columns= ["GnP weight %","electrical conductivity","percolation threshold"])
sig_results = df.to_csv(r'C:\Users\61432\Desktop\Codes\graphene nanocomposite modelling\graphene nanocomposite modelling\sig_result.csv',index=None,header=True)

print(percW*100)

