import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from gmri import buildMRI, buildgMRI
from gmri.orthogonalization_engine import orthogonalizationEngine

print(("Test on the 'gun' nonlinear eigenvalue problem from the NLEVP "
       "collection mantained by F. Tisseur."))
###############################################################################
allowed_tags = ["MRI", "gMRI"]
if len(sys.argv) > 1:
    approximation_strategy = sys.argv[1]
else:
    approximation_strategy = input(("Input approximation_strategy (allowed "
                                    "values: {})\n").format(allowed_tags))
if len(sys.argv) > 2:
    bisections = sys.argv[2]
else:
    bisections = input(("Input the desired number of bisections (the final "
                        "piecewise approximation will be supported on the "
                        "union of 2^bisections subintervals) or 'AUTO':\n"))
if len(sys.argv) > 3:
    eps_stab = float(sys.argv[3])
else:
    eps_stab = float(input("Input the desired stability tolerance:\n"))
###############################################################################
k2s, alphas = [1.25e4, 1.125e5], [0., 108.8774]
data = loadmat('nlevp/private/gun.mat')
data_mat = {key: csc_matrix(data[key]) for key in ["K", "M", "W1", "W2"]}
np.random.seed(42)
rhs = np.random.randn(data_mat["K"].shape[0])
lhs = lambda k2: (data_mat["K"] - k2 * data_mat["M"]
                + 1j * ((k2 - alphas[0] ** 2) ** .5 * data_mat["W1"]
                      + (k2 - alphas[1] ** 2) ** .5 * data_mat["W2"]))
###############################################################################

solve = lambda k2: spsolve(lhs(k2), rhs).reshape(-1, 1)
energyMatrix = data_mat["M"]
engine = orthogonalizationEngine(energyMatrix)

if approximation_strategy == "MRI":
    app = buildMRI(solve, energyMatrix, np.linspace(*k2s, 101),
                   eps_stab, subdivisions = 2 ** int(bisections))
elif approximation_strategy == "gMRI":
    app = buildgMRI(solve, energyMatrix, np.linspace(*k2s, 10001),
                    1e-3, eps_stab, bisections = bisections)
#poles = app.poles()
poles = [app_.poles() for app_ in app.apps] # only for coloring

### POSTPROCESS
k2test = np.linspace(*k2s, 1001)
u_app = app(k2test, None)
if isinstance(u_app, list):
    normApp = np.empty(len(k2test))
    for j, u_ in enumerate(u_app):
        normApp[j] = np.linalg.norm(u_)
else:
    normApp = np.linalg.norm(u_app, axis = 0)

k2stride = 40
k2testCoarse = k2test[::k2stride]
normEx = np.empty(len(k2testCoarse))
for j, k2 in enumerate(k2testCoarse):
    normEx[j] = engine.norm(solve(k2)).flatten()

### PLOTS
plt.figure()
plt.semilogy(k2testCoarse, normEx, 'o')
plt.semilogy(k2test, normApp)
for app_ in app.apps:
    plt.semilogy(app_.supp[0], [np.min(normApp)] * app_.supp.shape[1], 'x')
plt.legend(["Exact", "Approx"])
plt.xlabel("k"), plt.ylabel("||u(k)||"), plt.grid()
plt.show()

symbols = "x+.1234"
plt.figure()
for j, poles_ in enumerate(poles):
    plt.plot(np.real(poles_), np.imag(poles_), symbols[j % len(symbols)])
plt.xlabel("Re(poles)"), plt.ylabel("Im(poles)")
plt.xlim(k2s), plt.ylim(0, 5e4), plt.grid()
plt.show()

