# Load losses
from numpy import genfromtxt
losses = genfromtxt('graphs/nn_100_losses.csv', delimiter=',')

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
