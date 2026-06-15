 # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from math import pi

from pybemt.solver import Solver

df_exp = pd.read_csv("propeller_data.csv", delimiter=' ')

s = Solver('propeller.ini')

df, sections = s.run_sweep('v_inf', 20, 1, 44.0)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(df["J"],df["eta"],"r-",label="eta bemt")
ax.plot(df_exp["J"],df_exp["eta"],"r.",markersize=10,label="eta exp")

ax.legend(loc="center left")
ax.set_xlabel("J")
ax.set_ylabel("eta")


ax2=ax.twinx()
ax2.plot(df["J"],df["CT"],"b-",label="CT bemt")
ax2.plot(df_exp["J"],df_exp["CT"],"b.",markersize=10,label="CT exp")

ax2.plot(df["J"],df["CP"],"g-",label="CP bemt")
ax2.plot(df_exp["J"],df_exp["CP"],"g.",markersize=10,label="CP exp")

ax2.legend(loc="center right")
ax2.set_ylabel("CT & CP")
plt.show()



# df_exp = pd.read_csv("propeller_data.csv", delimiter=' ')

# ax = df.plot(x='J', y='eta', legend=None) 
# df_exp.plot(x='J',y='eta',style='C0o',ax=ax, legend=None)
# pl.legend(('BEMT, $\eta$','Exp, $\eta$'),loc='center left')

# pl.ylabel('$\eta$')
# ax2 = ax.twinx()
# pl.ylabel('$C_P, C_T$')

# df.plot(x='J', y='CP', style='C1-',ax=ax2, legend=None) 
# df_exp.plot(x='J',y='CP',style='C1o',ax=ax2, legend=None)

# df.plot(x='J', y='CT', style='C2-',ax=ax2, legend=None) 
# df_exp.plot(x='J',y='CT',style='C2o',ax=ax2, legend=None)


# pl.legend(('BEMT, $C_P$','Exp, $C_P$',
#     'BEMT, $C_T$','Exp, $C_T$'),loc='center right')

# pl.show()

