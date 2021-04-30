import numpy as np

class gbm():
    def __init__(self, x_0, mu, sigma, dt):
        self.x_0 = x_0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

        
    def initilize_run(self):
        self.x = [self.x_0]
        

    def step(self):
        dw = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=1)[0]
        dx = self.x[-1]*(self.mu*self.dt+self.sigma*dw)
        self.x.append(self.x[-1]+dx)
        
        
    def run(self, n_steps):
        for i in range(n_steps):
                self.step()
                

class rgbm():
    
    def __init__(self, x_0, mu, sigma, tau, dt, N):
        self.x_0 = x_0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.tau = tau
        self.N = N
    
        self.x = [np.array(N*[self.x_0])]
        

    def step(self):
        x_avg = np.mean(self.x[-1])
        dw = np.random.normal(loc=0, scale=np.sqrt(self.dt), size=self.N)
        dx = self.x[-1]*( (self.mu-self.tau)*self.dt + self.sigma*dw) + self.tau*x_avg*self.dt
        self.x.append(self.x[-1]+dx)
        
        
    def run(self, n_steps):
        for i in range(n_steps):
                self.step()
