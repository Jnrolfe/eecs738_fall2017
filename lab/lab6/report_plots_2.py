import matplotlib.pyplot as plt
import numpy as np

xs_1_r = [-2, 0, 2]
ys_1_r = [2, 8, 1]

xs_1_b = [-2, 0, 2]
ys_1_b = [-2, -1, -3]

supx_1 = [-2, 2, 0]
supy_1 = [2, 1, -1]

def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y, color='black')

graph('0.25 - (0.25)*x', range(-3, 4))

plt.scatter(supx_1, supy_1, color='black', s=100)
plt.scatter(xs_1_r, ys_1_r, color='red', s=50, label='+1 label')
plt.scatter(xs_1_b, ys_1_b, color='blue', s=50, label='-1 label')
plt.xlabel('X_{1} value')
plt.ylabel('X_{2} value')
plt.title('Separable Data')
plt.legend()
plt.show()
