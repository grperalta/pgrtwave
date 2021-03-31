"""
Prints the condition numbers of the system matrix in the hybrid formulation with
decreasing penalty parameters.
"""

from pgrtwave import *

def ocp_condition_numbers():
    print(strcolor.cyan_b("Penalty\tEig Max\t\tEig Min\t\tCond No"))
    for pen in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]:
        ocp = OCP(prm=dict(pen=pen, n=65, T=1.0, num_time_step=1000,
            hybrid=True, alpha=10.0, beta=1.0, gamma=1e-5, ocptol=1e-6,
            pressure_ave=True, velocity_ave=False, lagrange_ave=True))
        eigs_lm, eigs_sm, cond_no = sparse_matrix_condition_number(
            sp.diags(ocp.Mat.P_inv) * ocp.Mat.M_plus)
        print("{}\t{:.6e}\t{:.6e}\t{:.6e}".format(pen, eigs_lm, eigs_sm, cond_no))
    print(strcolor.cyan_b("\nSpatial Meshsize: ") + "\t{:.6e}".format(ocp.mesh.size()))
    print(strcolor.cyan_b("Temporal Stepsize:") + "\t{:.6e}".format(ocp.dt))


if __name__ == "__main__":
    print_start()
    ocp_condition_numbers()
    print_end()
