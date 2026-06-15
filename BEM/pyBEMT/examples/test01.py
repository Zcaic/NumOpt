from pybemt.airfoil import load_airfoil,Airfoil 
import matplotlib.pyplot as plt
import numpy as np

af=load_airfoil("CLARKY")

alpha=np.linspace(-180.0,180.0,100)
cl=af.Cl(alpha)
cd=af.Cd(alpha)

fig=plt.figure()
ax=fig.add_subplot(121)
ax.plot(alpha,cl,label="CL")
ax.legend()

ax=fig.add_subplot(122)
ax.plot(alpha,cd,label="CD")
ax.legend()

plt.show()