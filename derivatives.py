import numpy as np
import matplotlib.pyplot as plt

#sympy = symbolic math in python

import sympy as sym
import sympy.plotting.plot as symplot

# create symbolic variables in sympy
x = sym.symbols('x')

# create a function

fx = 2*x**2

# compute it's derivative

df = sym.diff(fx,x)

print(fx)
print(df)


# plot

# symplot(fx,(x,-4,4),title='the function')
# plt.savefig('function')

# symplot(df,(x,-4,4),title="it's derivative")
# plt.savefig('derivative')


# repeat with relu and sigmoid 

relu = sym.Max(0,x)
sigmoid = 1 / (1+sym.exp(-x))

# graph the fuctions

p = symplot(relu,(x,-4,4), label="ReLU", show=False, line_color="blue")
p.extend ( symplot(sigmoid,(x,-4,4), label="Sigmoid", show=False, line_color="red"))
p.legend = True
p.title = "The functions"
p.save("relu and sigmoid")


# plot theyr derivatives

p = symplot(sym.diff(relu), (x,-4,4), label = "df(ReLU)", show=False, line_color="blue")
p.extend(symplot(sym.diff(sigmoid), (x,-4,4), label = "df(Sigmoid)", show=False, line_color="red"))
p.legend = True
p.title = "The Derivatives"
p.save("df of relu and sigmoid")
