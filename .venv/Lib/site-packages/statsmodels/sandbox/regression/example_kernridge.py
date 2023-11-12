

import numpy as np
import matplotlib.pyplot as plt
from .kernridgeregress_class import GaussProcess, kernel_euclid


m,k = 50,4
upper = 6
scale = 10
xs = np.linspace(1,upper,m)[:,np.newaxis]
#xs1 = xs1a*np.ones((1,4)) + 1/(1.0+np.exp(np.random.randn(m,k)))
#xs1 /= np.std(xs1[::k,:],0)   # normalize scale, could use cov to normalize
##y1true = np.sum(np.sin(xs1)+np.sqrt(xs1),1)[:,np.newaxis]
xs1 = np.sin(xs)#[:,np.newaxis]
y1true = np.sum(xs1 + 0.01*np.sqrt(np.abs(xs1)),1)[:,np.newaxis]
y1 = y1true + 0.10 * np.random.randn(m,1)

stride = 3 #use only some points as trainig points e.g 2 means every 2nd
xstrain = xs1[::stride,:]
ystrain = y1[::stride,:]
xstrain = np.r_[xs1[:m/2,:], xs1[m/2+10:,:]]
ystrain = np.r_[y1[:m/2,:], y1[m/2+10:,:]]
index = np.hstack((np.arange(m/2), np.arange(m/2+10,m)))
gp1 = GaussProcess(xstrain, ystrain, kernel=kernel_euclid,
                   ridgecoeff=5*1e-4)
yhatr1 = gp1.predict(xs1)
plt.figure()
plt.plot(y1true, y1,'bo',y1true, yhatr1,'r.')
plt.title('euclid kernel: true y versus noisy y and estimated y')
plt.figure()
plt.plot(index,ystrain.ravel(),'bo-',y1true,'go-',yhatr1,'r.-')
plt.title('euclid kernel: true (green), noisy (blue) and estimated (red) '+
          'observations')
