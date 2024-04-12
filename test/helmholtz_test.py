import sys
import numpy as np
from matplotlib import pyplot as plt
from helmholtz_engine import solverHelmholtz1DFiniteDifferences
from gmri import buildMRI, buildgMRI
from gmri.orthogonalization_engine import orthogonalizationEngine

print(("Test on an FD-discretization of the Helmholtz equation on the interval"
       " [0, pi] with squared wavenumbers between 0 and 10^3. The domain is "
       "discretized using 10^3 grid points. The RHS is random."))
###############################################################################
allowed_tags = ["MRI", "gMRI"]
if len(sys.argv) > 1:
    approximation_strategy = sys.argv[1]
else:
    approximation_strategy = input(("Input approximation_strategy (allowed "
                                    "values: {})\n").format(allowed_tags))
if len(sys.argv) > 2:
    bisections = int(sys.argv[2])
else:
    bisections = int(input(("Input the desired number of bisections (the "
                            "final piecewise approximation will be supported "
                            "on the union of 2^bisections subintervals):\n")))
###############################################################################
k2s = [0., 1000.]

domain_length = np.pi
n_mesh = 1000
###############################################################################

solver = solverHelmholtz1DFiniteDifferences(domain_length, n_mesh)
energyMatrix = solver.energyMatrix(0.)
engine = orthogonalizationEngine(energyMatrix)

if approximation_strategy == "MRI":
    appList = buildMRI(solver.solve, energyMatrix, np.linspace(*k2s, 101),
                       subdivisions = 2 ** bisections)
elif approximation_strategy == "gMRI":
    appList = buildgMRI(solver.solve, energyMatrix, np.linspace(*k2s, 10001),
                        1e-3, bisections = bisections)
k2sGrid = [k2s[0]] + [np.max(app.supp) for app in appList]
poles = [app.poles() for app in appList]

### POSTPROCESS
poles_int_range = [np.ceil(max(1., k2s[0])) ** .5 * (domain_length / np.pi),
                   np.floor(max(1., k2s[-1])) ** .5 * (domain_length / np.pi) + 1]
polesEx = (np.pi * np.arange(*poles_int_range) / domain_length) ** 2

k2test = np.linspace(*k2s, 1001)
normApp = np.empty(len(k2test))
k2stride = 25
k2testCoarse = k2test[::k2stride]
normEx = np.empty(len(k2testCoarse))
    
idxGrid = 0 # index of approximation that should be used
for j, k2 in enumerate(k2test):
    while idxGrid < len(k2sGrid) - 1 and k2 > k2sGrid[idxGrid + 1]:
        idxGrid += 1
    normApp[j] = np.linalg.norm(appList[idxGrid](k2, None)[:, 0])

for j, k2 in enumerate(k2testCoarse):
    normEx[j] = engine.norm(solver.solve(k2)).flatten()

### PLOTS
plt.figure()
plt.semilogy(k2testCoarse, normEx, 'o')
plt.semilogy(k2test, normApp)
for app in appList:
    plt.semilogy(app.supp[0], [np.min(normApp)] * app.supp.shape[1], 'x')
plt.legend(["Exact", "Approx"])
plt.xlabel("k"), plt.ylabel("||u(k)||"), plt.grid()
plt.show()

symbols = "x+.1234"
plt.figure()
plt.plot(polesEx, [0] * len(polesEx), 'o')
for j, poles_ in enumerate(poles):
    plt.plot(np.real(poles_), np.imag(poles_), symbols[j % len(symbols)])
plt.legend(["Exact"])
plt.xlabel("Re(poles)"), plt.ylabel("Im(poles)")
plt.xlim(k2s), plt.ylim(-.5, .5), plt.grid()
plt.show()

