# The python code to generate the data for transverse field Ising model (TFIM)

import numpy as np
import sys

# TenPy
from tenpy import TFIChain
from tenpy.algorithms import dmrg, tebd
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.networks.site import SpinHalfSite
from tenpy.models.tf_ising import TFIChain


def getGsMPS(M, m, chi):
    # Find lowest #m MPS with bond dimension 2 as initial states
    # Also find #m lowest states with large bond dimensions for reference

    print('Find GS, bond dimension =', chi)
    prev_MPS = []
    energies = []
    trunc_err = 1e-1 if chi == 2 else 1e-5
    pstate = [['up']]
    if M.lat.dim == 2:
        pstate = [pstate]
    for i in range(m):
        psi = MPS.from_lat_product_state(M.lat, pstate, dtype=complex)
        dmrg_params = {
            'mixer': None,  # setting this to True helps to escape local minima
            'max_E_err': 1.e-10,
            'trunc_params': {
                'chi_max': chi,
                'svd_min': 1.e-10,
            },
            'max_trunc_err': trunc_err
        }
        eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
        if len(prev_MPS) > 0:
            eng.init_env(orthogonal_to=prev_MPS)
        E, psi = eng.run()  # the main work; modifies psi in place
        psi.canonical_form()
        print('MPS', i, ': energy', E)
        prev_MPS.append(psi)
        energies.append(E)
    return prev_MPS, energies


def computeWMat(model, obs, obs_names, m, N_sample, epsilon, noise_level=1e-3, chi=50):
    # Main part: compute the M matrices
    T = 0.5 / epsilon
    np.random.seed(0)
    L = model.lat.Ls[0]

    # Draw times
    t = np.random.normal(loc=0.0, scale=T, size=N_sample)
    t = np.where(np.abs(t) > 3 * T, 0, t)  # Hard cutoff in evolution time
    t = np.sort(t)

    # Separate to positive / negative time evolutions
    pos_zero = np.argmax(t >= 0)
    neg_t = t[:pos_zero]
    pos_t = t[pos_zero:]
    neg_t = neg_t[::-1]

    # Min time step
    dt = 0.05

    print()
    print('t', t)
    print('Generate Mk')
    print('chi', chi)

    # Get ground states
    print('Get ground states!')
    gsMPSs, gsEs = getGsMPS(model, m, chi)

    # Build the sensing matrix Φ
    print('Get initial MPS of bond dimension 2')
    Φ, ΦEs = getGsMPS(model, m, 2)

    print('Overlaps with m lowest energy eigenstates')
    for psi in Φ:
        print([psi.overlap(gs) for gs in gsMPSs])
        for ob_idx, ob in enumerate(obs):
            psip = psi.copy()
            ob.apply_naively(psip)
            print('Initial observable val of',
                  obs_names[ob_idx], psi.overlap(psip) / L / L)

    # Generate sample matrices M^(k)
    tebd_params = {
        'order': 2,
        'max_trunc_err': 1.e-6,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10
        },
    }

    # Store evolved states
    evolvedStates = [[] for _ in range(m)]

    for i in range(m):
        print('m', i)

        # Negative time
        tmp_psi = Φ[i].copy()
        eng = tebd.TEBDEngine(tmp_psi, model, tebd_params)
        # eng = tdvp.TwoSiteTDVPEngine(tmp_psi, model, tdvp_params)
        for k in range(pos_zero):
            dtk = neg_t[0] if k == 0 else neg_t[k] - neg_t[k-1]
            dtk = - dtk
            eng.run_evolution(int(dtk / dt), -dt)
            eng.run_evolution(1, - (dtk - dt * int(dtk / dt)))
            # tmp_psi.canonical_form()
            if (k % 10 == 0 or k == pos_zero - 1):
                print('t =', neg_t[k], ', trunc error =',  eng.trunc_err_bonds[L//2].eps,
                      ', overlap with ini state =', [tmp_psi.overlap(Φ[j]) for j in range(m)])
            evolvedStates[i].insert(0, tmp_psi.copy())
        print([len(evolvedStates[j]) for j in range(m)], 'evolved states')

        # Positive time
        tmp_psi = Φ[i].copy()
        eng = tebd.TEBDEngine(tmp_psi, model, tebd_params)
        # eng = tdvp.TwoSiteTDVPEngine(tmp_psi, model, tdvp_params)
        for k in range(N_sample - pos_zero):
            dtk = pos_t[0] if k == 0 else pos_t[k] - pos_t[k-1]
            eng.run_evolution(int(dtk / dt), dt)
            eng.run_evolution(1, (dtk - dt * int(dtk / dt)))
            # tmp_psi.canonical_form()
            if (k % 10 == 0 or k == N_sample - pos_zero - 1):
                print('t =', pos_t[k], ', trunc error =',
                      eng.trunc_err_bonds[L//2].eps)
            evolvedStates[i].append(tmp_psi.copy())
        print([len(evolvedStates[j]) for j in range(m)], 'evolved states')

    print()
    print('Get overlaps')
    M_list = [[np.zeros((m, m), dtype=complex)
               for _ in range(N_sample)] for _ in range(N_sample)]
    M_O_list = [[[np.zeros((m, m), dtype=complex) for _ in range(
        N_sample)] for _ in range(N_sample)] for _ in range(len(obs))]

    for k in range(N_sample):
        for l in range(N_sample):
            if (k % 10 == 0 and l % 10 == 0):
                print(f'k = {k}, l = {l}')
            for i in range(m):
                for j in range(m):
                    M_list[k][l][i, j] = evolvedStates[i][k].overlap(
                        evolvedStates[j][l])
                    for ob_idx, ob in enumerate(obs):
                        psip = evolvedStates[j][l].copy()
                        ob.apply_naively(psip)
                        M_O_list[ob_idx][k][l][i, j] = evolvedStates[i][k].overlap(
                            psip)
                        if obs_names[ob_idx] == 'Sx2':
                            M_O_list[ob_idx][k][l][i,
                                                   j] = M_O_list[ob_idx][k][l][i, j] / L / L

    return t, M_list, M_O_list, gsEs


# Parameters
if (len(sys.argv) != 8):
    print('Usage: python NonOrthogonalQMEGS_MPS.py <g> <L> <m> <N_sample> <epsilon> <chi> <noise_level>')
    sys.exit(1)
L = int(sys.argv[1])                # system size
g = float(sys.argv[2])              # transverse field strength
m = int(sys.argv[3])                # number of MPS to sample
N_sample = int(sys.argv[4])         # number of time samples
epsilon = float(sys.argv[5])        # inverse Gaussian width
chi = int(sys.argv[6])              # bond dimension for MPS
noise_level = float(sys.argv[7])    # noise level for M matrices

model_params = {
    'J': 1.,
    'g': g,
    'L': L,
    'bc_MPS': 'finite',
    'bc_x': 'open',
    'conserve': None
}

model = TFIChain(model_params)

# Observable: (Sum Sx)^2
site = SpinHalfSite(conserve='None', sort_charge=False)
Sx, Id = site.Sx, site.Id


gridl = [[Id, Sx]]
grid = [[Id, Sx], [None, Id]]
gridr = [[Sx], [Id]]
grids = [gridl] + [grid] * (L - 2) + [gridr]
sx = MPO.from_grids(model.lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)

gridl2 = [[Id, Sx * np.sqrt(2), Id]]
grid2 = [[Id, Sx * np.sqrt(2), Id], [None, Id, Sx *
                                     np.sqrt(2)], [None, None, Id]]
gridr2 = [[Id], [Sx * np.sqrt(2)], [Id]]
grids2 = [gridl2] + [grid2] * (L - 2) + [gridr2]
sx2 = MPO.from_grids(model.lat.mps_sites(), grids2, bc='finite', IdL=0, IdR=-1)

obs = [sx, sx2]
obs_names = ['Sx', 'Sx2']

t, M_list, M_O_list, gsEs = computeWMat(
    model, obs, obs_names, m, N_sample, epsilon, noise_level, chi)

# Save results
fname = f'data/tfi_chain_g{g}_L{L}_chi{chi}_N{N_sample}'
np.savetxt(f'{fname}_ts', t)
np.savetxt(f'{fname}_gsEs', gsEs)
np.savetxt(f'{fname}_M_list', np.array(
    M_list).reshape(N_sample * N_sample, m * m))

for ob_idx, ob in enumerate(obs):
    np.savetxt(f'{fname}_M_{obs_names[ob_idx]}_list', np.array(
        M_O_list[ob_idx]).reshape(N_sample * N_sample, m * m))
