import numpy as np
import matplotlib.pyplot as plt

from gbm_rgbm import gbm

x_0 = 0.1
mu = 0.2
dt = 0.1
sigma = 0.3

N=10
n_steps=1000


model=gbm(x_0, mu, sigma, dt)

res=[]
for i in range(N):
    model.initilize_run()
    model.run(n_steps)
    res.append(model.x)


plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
for x_n in res:
    plt.plot(x_n)
plt.title('geometric Brownian motion realizations')

plt.subplot(1,2,2)
for x_n in res:
    plt.plot(x_n)
plt.yscale('log')
plt.title('geometric Brownian motion realizations - log scale')

plt.savefig("gbm.png")
plt.close()
#plt.show()
