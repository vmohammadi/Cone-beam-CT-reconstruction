# -*- coding: utf-8 -*-
"""
Program for CT calibration


ex, ey, ez :        x, y , z directions of the coordiantion system
R :                 the distance from the cone vertex to the rotation axis. It therefore 
defines the vertex position in the (x, y, z) system as (R, 0, 0).
SDD :               the shortest distance from the source to the detector plane
D :                 distance of the source to the center of rotation
ew :                the unit vector orthogonal to the detector plane
θ ∈ [−π/2, π/2] and φ ∈ [−π/2, π/2] parametrize the unit vector ew
ew :                = (cos θ cos φ, cos θ sin φ, sin θ)
θ = 0 :             is assured when the turntable is horizontal while the detector plane is vertical.
η ∈ [−π/2, π/2) :   used to define two orthogonal unit directions, eu and ev
eu and ev :         along which detector pixels are aligned in the detector plane
α :                 = (−sin φ, cos φ, 0)
β :                 = (−sin θ cos φ,−sin θ sin φ, cos θ)
eu :                = cos ηα + sin ηβ
ev :                = cos ηβ − sin ηα
u0, v0 :            distances separating the detector pixel of least u and v values 
                    from the orthogonal projection of the cone vertex onto the detector plane
r :                 half of horizontal distance of the point objects [mm]
h :                 vertical distance of the point objects [mm]
ubar, vbar          center of elipse

-------------------- Attention -----------------------------------------------

In pointobj function, the threshold needs to be redefined, because the threshold 
value depends on the conditions of your image acquisition.Also, after the threshold, 
we break the image to two parts, each part having only one ball. Be sure that only 
the balls are in the iamges and the cylindrical support is not in the image.
"""

import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import os
from os.path import join
import cv2 as cv
import math

#%% Functions 

def pointobj (img):
    """
    pointobj gets the image, changes it to the image complement. 
    Then finds the point objects and calculates the centers of them.
    """
    imcom = np.invert(img) # invert the image
    ret, imth = cv.threshold(imcom, 50000, 66600, cv.THRESH_BINARY) # Threhsold and keep the point objects
    imgTop = imth[0:256,0:512] # Top part of image including the top point object
    imgDown = imth[256:256+190,0:512] # Down part of image including the down point object
    
    positions = np.nonzero(imgTop) # Get the box address around the point object, to calculate the center point
    top = positions[0].min()
    bottom = positions[0].max()
    vt1 = ((top + bottom)/2) 
    vt1 = (512 - vt1) * detector_pixel_size # Because (0,0) is considered as the left down corner of the image
    left = positions[1].min()
    right = positions[1].max()
    ut1 = ((left + right )/2) * detector_pixel_size
    
    positions = np.nonzero(imgDown) # Get the box address around the point object, to calcumalte the center point
    top = positions[0].min() + 256 # Because we already divided the image to two parts, we add 256 to the values
    bottom = positions[0].max() + 256
    vt2 = ((top + bottom)/2)
    vt2 = (512 - vt2)  * detector_pixel_size
    left = positions[1].min()
    right = positions[1].max()
    ut2 = ((left + right )/2) * detector_pixel_size
    
    # print (ut1, vt1, ut2, vt2)
    return ut1, vt1, ut2, vt2
    
#%% Calculation of η
# the projections of two point objects, lead to two circles on the image plane.
# goal is to find the centers of those two circles for obtanning η

N = 360 # Normally N = 5 is enough but the bigger the N, the better results
detector_pixel_size = 0.06836  # [mm]
r = 7.5 # Half of horizontal distance of the point objects [mm]
h = 19 # Vertical distance of the point objects [mm]
SDD = 294.63895

listim = os.listdir('Calib')
u1 = np.array([]); 
v1 = np.array([]); 
u2 = np.array([]); 
v2 = np.array([]); 
Atop = np.array([0,0]); 
Ctop = np.array([0]);
Adown = np.array([0,0]); 
Cdown = np.array([0]);

for i in range (359):
    img = mpimg.imread(join('Calib', 'frame_{:01d}.0.tif'.format(i))) # Read the image
    img[img==0] = 60200 # Dead pixels which are zero should be replaced, if not, they produce errors
    ut1, vt1, ut2, vt2 = pointobj(img)
    
    u1 = np.append (u1, ut1) # These are coordinates of point objects in each frame
    v1 = np.append (v1, vt1)
    u2 = np.append (u2, ut2)
    v2 = np.append (v2, vt2)
    # points = np.vstack((u1,v1,u2,v2))


for i in range(179):
    A1 = np.array([u1[i+180]-u1[i], -v1[i+180]+v1[i]])
    C1 = np.array([v1[i]*u1[i+180]-v1[i+180]*u1[i]])
    Atop = np.vstack((Atop, A1))
    Ctop = np.vstack((Ctop, C1))
    # As we have two k's, then we will have two relations, we add these two relations and make a matrix of n*4 constants
    # Then we will have a matrix of 4*1 variables (v1 u1 v2 u2), and for the response matrix, we will have n*1
    A2 = np.array([u2[i+180]-u2[i], -v2[i+180]+v2[i]])
    C2 = np.array([v2[i]*u2[i+180]-v2[i+180]*u2[i]])
    Adown = np.vstack((Adown, A2))
    Cdown = np.vstack((Cdown, C2))


# We will have equations with two variables, uhat and vhat, that need to be determined, once for 
# top point object, once for down point object    
Atop = Atop[1:180]    
Ctop = Ctop[1:180]    
B1 = np.linalg.lstsq(Atop,Ctop) # We use least square solution
B1 = B1[0]
vhat1 = B1[0]; uhat1 = B1[1];
Adown = Adown[1:180]    
Cdown = Cdown[1:180]  
B2 = np.linalg.lstsq(Adown,Cdown) # We use least square solution
B2 = B2[0]
vhat2 = B2[0]; uhat2 = B2[1];   
        
η = np.arctan((uhat1 - uhat2)/(vhat1-vhat2))
print ('η is equal to : ', η)
del Atop, Ctop, Adown, Cdown, B1, B2
#%% Determination of elipses equations

Xmat1 = np.array([0,0,0,0,0])
Res1 = np.array([0])
Xmat2 = np.array([0,0,0,0,0])
Res2 = np.array([0])

for i in range (179):
    Xt1 = np.array([u1[i]**2, -2*u1[i], -2*v1[i], 2*u1[i]*v1[i], 1]) # Matrix of known variables
    Xmat1 = np.vstack((Xmat1,Xt1))
    Rt1 = np.array([-(v1[i]**2)])
    Res1 = np.vstack((Res1,Rt1))
    
Xmat1 = Xmat1[1:179]
Res1 = Res1[1:179]
Pmat1 = np.linalg.lstsq(Xmat1,Res1) # Least square solution # Pmat is 5 by 1
Pmat1 = Pmat1[0]

ubar1 = (Pmat1[1]-Pmat1[2]*Pmat1[3])/(Pmat1[0]-Pmat1[3]**2)
vbar1 = (Pmat1[0]*Pmat1[2] - Pmat1[1]*Pmat1[3])/(Pmat1[0]-Pmat1[3]**2)
a1 = Pmat1[0]/(Pmat1[0]*(ubar1**2) + (vbar1**2) + 2*Pmat1[3]*ubar1*vbar1 - Pmat1[4])
b1 = a1/Pmat1[0]
c1 = Pmat1[3]*b1


for i in range (179):
    Xt2 = np.array([u2[i]**2, -2*u2[i], -2*v2[i], 2*u2[i]*v2[i], 1])
    Xmat2 = np.vstack((Xmat2,Xt2))
    Rt2 = np.array([-(v2[i]**2)])
    Res2 = np.vstack((Res2,Rt2))
    
Xmat2 = Xmat2[1:179]
Res2 = Res2[1:179]
Pmat2 = np.linalg.lstsq(Xmat2,Res2) # Pmat is 5 by 1
Pmat2 = Pmat2[0]

ubar2 = (Pmat2[1]-Pmat2[2]*Pmat2[3])/(Pmat2[0]-Pmat2[3]**2)
vbar2 = (Pmat2[0]*Pmat2[2] - Pmat2[1]*Pmat2[3])/(Pmat2[0]-Pmat2[3]**2)
a2 = Pmat2[0]/(Pmat2[0]*(ubar2**2) + (vbar2**2) + 2*Pmat2[3]*ubar2*vbar2 - Pmat2[4])
b2 = a2/Pmat2[0]
c2 = Pmat2[3]*b2
    
print ('center of elipse 1 is (in mm): ', ubar1,',',vbar1)
print ('center of elipse 2 is (in mm): ', ubar2,',',vbar2)

del Xmat1, Xmat2, Res1, Res2, Pmat1, Pmat2
#%% Calculation of source to center of rotation distance

vP901 = v1[90] # We use the image in 90° and 180° to have the best precision
vP1801 = v1[180]
vP902 = v2[90]
vP1802 = v2[180]
ha = vP901 - vP902
hb = vP1801 - vP1802

D = ((SDD*h)/ha)-r
print ('source to center of rotation distance is : ', D)
#%% Calculation of scanner parameters
m0 = (vbar2 - vbar1) * math.sqrt(b2 - (c2**2/a2))
m1 = math.sqrt(np.abs(b2 - (c2**2/a2)))/ math.sqrt(np.abs(b1 - (c1**2/a1)))
n0 = (1 - m0**2 - m1**2)/(2 * m0 * m1)
n1 = (a2 - a1 * m1**2)/(2 * m0 * m1)
E = 1 # Epsilon
sumN = 0
sign1 = -1 # the sign of zk can be easily obtained by data inspection, e.g. when zk > 0, one
# observes that the projection of the point object moves clockwise in the detector plane for a
# counterclockwise rotation of the turntable
sign2 = +1

vstar0 = vhat1 - sign1 * math.sqrt(np.abs(a1 + a1**2 * D**2))/math.sqrt(np.abs(a1 * b1 - c1**2))
ustar0 = (ubar1/2) + (ubar2/2) + (c1/2*a1) * (vbar1 - vstar0) + (c2/2*a2) * (vbar2 - vstar0)
ro1 = math.sqrt (np.abs(a1 * b1 - c1**2)) / math.sqrt(np.abs(a1 * b1 + a1**2 * b1 * D**2 - c1**2))
ro2 = math.sqrt (np.abs(a2 * b2 - c2**2)) / math.sqrt(np.abs(a2 * b2 + a2**2 * b2 * D**2 - c2**2))
zi1 = (D * sign1 * a1 * math.sqrt(a1)) / math.sqrt(np.abs(a1 * b1 + a1**2 * b1 * D**2 - c1**2))
zi2 = (D * sign2 * a2 * math.sqrt(a2)) / math.sqrt(np.abs(a2 * b2 + a2**2 * b2 * D**2 - c2**2))
v0 = v1 - sign1 * math.sqrt(np.abs(a1 + a1**2 * D**2))/math.sqrt(np.abs(a1 * b1 - c1**2))
u0 = ubar1 + (c1/a1 * (vbar1 - v0))
phi = math.asin((-c1 * zi1/2 * a1) - (c2 * zi2/2 * a2))

ustar01 = ubar1 - (D * math.sin(phi) * math.cos(phi))/(math.cos(phi)**2 - ro1**2)
vstar01 = vbar1 - (D * zi1 * math.cos(phi))/(math.cos(phi)**2 * ro1**2)

print('Phi angle is : ', phi)
#%% Determination of offset of center of rotation
# This is for the determination of how much the center of rotation is far from
# the line that connects the source to the center of the detector [pixels]

img = mpimg.imread(join('Calib', 'frame_{:01d}.0.tif'.format(0))) # Read the image
img[img==0] = 60200 # Dead pixels which are zero should be replaced, if not, they produce errors
ut1, vt1, ut2, vt2 = pointobj(img)
cent_bille = np.divide((ut2 + ut1),2)/detector_pixel_size
if cent_bille == 256:
    print("No horizontal center offset")
elif cent_bille <256:
    dis = cent_bille - 256
    print("Offset of center of rotation is : ",dis)
else:
    dis = cent_bille - 256
    print("Offset of center of rotation is : ",dis)

