import matplotlib.pyplot as plt
import numpy as np
import practice as p
from scipy.optimize import curve_fit

data = np.loadtxt("chainvalues.txt", delimiter=",")
print(data.shape)

# Paramters of MCMC run
nwalkers = 500  # No. of chains/walkers to be used
iterations = 300  # No. of iterations to be used

# Scatter plot
print("Plotting scatter plot")
distancechain = data[:500, 300 - 1]
inclinationchain = data[:500, 600 - 1]
lik = data[:500, 1200 - 1]
plt.scatter(distancechain, inclinationchain, c=lik)
plt.xlabel("Distance (Mpc)")
plt.ylabel("Inclination angle (Radians)")
cbar = plt.colorbar()
cbar.set_label("loglikelihood")
plt.show()

# Plots of MCMC Chains vs iterations
print("Plotting chain  values vs iterations")
distancechain = data[:500, :300]
# for emcee it would be 2d and for emceePT it would be 3d. Here we are taking 0 temp
print(" Nwalkers * Ninterations ", distancechain.shape)
x = np.arange(iterations)
for i in range(nwalkers):
    plt.plot(x, distancechain[i, :], label=f"X walker #{str(i)}")
plt.xlabel("iterations")
plt.ylabel("Values attained by chains")
plt.legend()
plt.show()


# # Plots of final values of nwalkers
binsize = 25  # bin size for the histogram of final values of the walkers
# Rows * Columns = Nwalkers * Iterations, Therfore last iteration is the final value of the chains/walkers
distancechain = data[:500, :300]
# Histogram of the last values of the chains
n, bins, patches = plt.hist(distancechain[:, -1], bins=binsize, density=True)
print("Mean and std dev of the last values of chain are ", np.mean(distancechain[:, -1]), np.std(distancechain[:, -1]))

bins_central = 0.5 * (bins[1:] + bins[:-1])
popt, pcov = curve_fit(p.gauss, bins_central, n)
y = p.gauss(bins_central, popt[0], popt[1])
plt.plot(bins_central, y, lw=3, marker="o", label="Gaussian")
mean_fit = format(popt[0], ".4f")
Std_fit = format(popt[1], ".4f")
print("Mean and Std dev of the fit is", mean_fit, Std_fit)

# y = stats.chi.pdf(bins_central, df = 1)
# plt.plot(bins_central, y, lw=3, marker='o', label = "Chi square with 1 dof")
plt.legend()
plt.xlabel("Final values of chains")
plt.ylabel("PDF")
plt.show()
