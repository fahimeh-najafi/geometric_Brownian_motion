import numpy as np
import matplotlib.pyplot as plt

from gbm_rgbm import rgbm


x_0  = 0.1
mu = 0.2
dt = 0.1
sigma = 0.3
tau =  0.1
N = 10
n_steps = 1000

model=rgbm(x_0, mu, sigma, tau, dt, N)
model.run(n_steps)
res = model.x

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
for x_n in np.array(res).T:
    plt.plot(x_n)
plt.title('reallocating geometric Brownian motion realizations')

plt.subplot(1,2,2)
for x_n in np.array(res).T:
    plt.plot(x_n)
plt.yscale('log')
plt.title('reallocating geometric Brownian motion realizations - log scale')

plt.show()
