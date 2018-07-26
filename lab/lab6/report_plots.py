import matplotlib.pyplot as plt
import numpy as np

xs_2_r = [8, 5, -5, 7, 5]
ys_2_r = [2, -1, 1, 1, 2]

xs_2_b = [2, -5, -5, 6]
ys_2_b = [10, 0, 2, 3]

slackx_1 = [-5, 6]
slacky_1 = [1, 3]

def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y, color='black')

graph('x+1', range(-5, 10))
plt.scatter(slackx_1, slacky_1, color='black', s=100)
plt.scatter(xs_2_r, ys_2_r, color='red', label='-1 label', s=50)
plt.scatter(xs_2_b, ys_2_b, color='blue', label='+1 label', s=50)
plt.xlabel('X_{1} value')
plt.ylabel('X_{2} value')
plt.title('Inseparable Data')
plt.legend()
plt.show()
