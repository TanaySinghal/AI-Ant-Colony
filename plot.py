# Load losses
import numpy as np
import math
from numpy import genfromtxt
losses = genfromtxt('graphs/nn_50_losses.csv', delimiter=',')

# sum_l = 0
# new_N = math.floor(len(losses)/10)
# new_losses = np.zeros(new_N)
# for i in range(len(losses)):
#   sum_l += losses[i]
#   if i % 10 == 9:
#     new_losses[math.floor(i/10)] = sum_l / 10
#     sum_l = 0

# Graph loss
print("PLOTTING")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()

plt.title("TD NN Loss")
plt.xlabel("# Games")
plt.ylabel("Loss")
plt.ylim(0,1)
plt.plot(losses)
plt.show()
