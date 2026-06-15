 # -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from math import pi


from pybemt.solver import Solver

s = Solver('tmotor28.ini')

df, sections = s.run_sweep('rpm', 20, 1000.0, 3200.0)

# ax = df.plot(x='rpm', y='T') 
# ax2 = df.plot(x='rpm', y='P') 

df_exp = pd.read_csv("tmotor28_data.csv", delimiter=';')

with plt.style.context(["science","no-latex"]):
    fig=plt.figure(figsize=(16,8))
    ax=fig.add_subplot(121)
    ax.plot(df["rpm"],df["T"],"-",label="bemt")
    ax.plot(df_exp["RPM"],df_exp["T(N)"],"o",label="exp")
    ax.set_xlabel("RPM")
    ax.set_ylabel("Trust(N)")
    ax.legend()

    ax=fig.add_subplot(122)
    ax.plot(df["rpm"],df["P"],"-",label="bemt")
    ax.plot(df_exp["RPM"],df_exp["P(W)"],"o",label="exp")
    ax.set_xlabel("RPM")
    ax.set_ylabel("P(W)")
    ax.legend()

    plt.tight_layout()
    plt.show()

# df_exp.plot(x='RPM',y='T(N)',style='o',ax=ax)
# pl.figure(1)
# pl.ylabel('Thrust (N)')
# pl.legend(('BEMT','Experiment'))
# df_exp.plot(x='RPM',y='P(W)',style='o',ax=ax2)
# pl.figure(2)
# pl.ylabel('Power (W)')
# pl.legend(('BEMT','Experiment'))


# pl.show()

