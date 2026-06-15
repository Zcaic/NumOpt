import matplotlib.pyplot as plt 
from pathlib import Path
import numpy as np

# def read_airfoil_data(file_path):
#     x=[[],[]]
#     y=[[],[]]
#     with open(file_path,"r") as f:
#         lines = f.readlines()
#         index = 0
#         for line in lines:
#             line = line.strip()
#             if line:
#                 xi,yi = map(float,line.strip().split())
#                 x[index].append(xi)
#                 y[index].append(yi)
#             else:
#                 index += 1
#     x[0].reverse()
#     y[0].reverse()
#     return x,y

# 翼型数据处理
# x,y = read_airfoil_data("naca0012.dat")
# x_up = np.array(x[0])
# x_low = np.array(x[1])
# ksai_TE = 0.01

x_up = np.linspace(1,0,100)
x_low = np.linspace(0,1,100)
ksai_TE = 0.01


# CST:
#递归函数：
def factorial(n):
    if n==0 or n==1:
        return 1
    else :
        return (n*factorial(n-1))

#类函数
def C(x,N1,N2):   
        c = (x**N1)*((1-x)**N2)
        return c

#形函数
def S_up(x,Au,order):
    Su=0.0
    for i in range(order+1):
        K = factorial(order)/(factorial(i)*factorial(order-i))
        Si = K*x**i*(1-x)**(order-i)
        Su += Au[i]*Si
        # Su =+ Su 
    return Su

def S_low(x,Al,order):
    Sl=0
    for i in range(order+1):
        K = factorial(order)/(factorial(i)*factorial(order-i))
        Si = K*x**i*(1-x)**(order-i)
        Sl += Al[i]*Si
        # Sl =+ Sl 
    return Sl

Au = [0.16973543,0.15195591,0.1393507,0.13933047]        # order=3
Al = [-0.16973543,-0.15195591,-0.1393507,-0.13933047]         
y_up = np.array(C(x_up,0.5,1)*S_up(x_up,Au,3)+x_up*ksai_TE)
y_low = np.array(C(x_low,0.5,1)*S_low(x_low,Al,3)+x_low*-ksai_TE)
x = np.hstack((x_low,x_up))
y = np.hstack((y_low,y_up))

plt.figure(figsize=(6,4))
plt.plot(x,y,label="naca_63415_para")
plt.axis("equal")
plt.title("NACA_63415 AirFoil")
plt.grid("True")
plt.legend()
plt.show()