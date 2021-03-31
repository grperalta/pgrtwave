"""
Computes the experimental orders of convergence in the hybrid method. (Example 3)
"""

from pgrtwave import *

def hybrid_error_table(_type="space", full_info=True):
    """
    Prints the table of errors and orders of convergence for the optimal controls,
    states, and adjoint states in the hybrid formulation.

    Spatial errors if <_type="space"> and temporal errors if <_type="time">.
    """

    if _type == "space":
        data = [9, 17, 33, 65, 129]
    elif _type == "time":
        data = [4, 8, 16, 32, 64]
    txt = "Control\t\tVelocity\tPressure\tAdj Velocity\tAdj Pressure"
    row_data = "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}"

    error = np.zeros((len(data), 6)).astype(float)
    order = np.zeros((len(data)-1, 5)).astype(float)
    eig_max = []
    eig_min = []
    cond_no = []

    i = 0
    for num in data:
        if _type == "space":
            ocp = OCP(prm=dict(n=num, T=1.0, num_time_step=1000, hybrid=True,
                alpha=1.0, beta=1.0, gamma=1.0, ocptol=1e-6, pressure_ave=False,
                    velocity_ave=False, lagrange_ave=False))
        elif _type == "time":
            ocp = OCP(prm=dict(n=129, T=1.0, num_time_step=num, hybrid=True,
                alpha=1.0, beta=1.0, gamma=1.0, ocptol=1e-6, pressure_ave=False,
                    velocity_ave=False, lagrange_ave=False))
        print_details(ocp.mesh, ocp.Mat, ocp.prm, ocp.dt)

        # x and y coordinates of nodes in the mesh
        x = ocp.mesh.node[ocp.mesh.tri-1, 0].reshape(3*ocp.mesh.num_elems,)
        y = ocp.mesh.node[ocp.mesh.tri-1, 1].reshape(3*ocp.mesh.num_elems,)

        # construct exact control, state and adjoint state
        adjoint_exact \
            = Wave_Vars(exact_dual_p(ocp.t, x, y),
            exact_dual_v_hybrid(ocp.t, ocp.mesh, ocp.transformations))
        state_exact \
            = Wave_Vars(desired_p(ocp.t, x, y),
            desired_v_hybrid(ocp.t, ocp.mesh, ocp.transformations))
        control_exact = - adjoint_exact.pre / ocp.prm["gamma"]

        # construct right hand side and desired state
        ocp.rhs = exact_source(ocp.t, x, y) - control_exact
        ocp.desired \
            = Wave_Vars(state_exact.pre - exact_dual_source(ocp.t, x, y)
            / ocp.prm["alpha"], state_exact.vel)

        # solve optimal control problem and post-processing
        sol = Barzilai_Borwein(ocp, SecondPoint=None, info=full_info, version=3)
        sol["state"].pre = HFEM_PostProcess(sol["state"], ocp.Mat, ocp.mesh)
        sol["adjoint"].pre = HFEM_PostProcess(sol["adjoint"], ocp.Mat, ocp.mesh)
        #sol["control"] =  sol["adjoint"].pre * (-1 / ocp.prm["gamma"])

        # error in control
        error[i, 0] = pre_norm_P1(control_exact - sol["control"],
            ocp.area, ocp.area_kron, ocp.dt, ocp.Mat)
        # error in velocity
        error[i, 1] = vel_norm(state_exact.vel - sol["state"].vel,
            ocp.dt, ocp.Mat)
        # error in pressure
        error[i, 2] = pre_norm_P1(state_exact.pre - sol["state"].pre,
            ocp.area, ocp.area_kron, ocp.dt, ocp.Mat)
        # error in dual velocity
        error[i, 3] = vel_norm(adjoint_exact.vel - sol["adjoint"].vel,
            ocp.dt, ocp.Mat)
        # error in dual pressure
        error[i, 4] = pre_norm_P1(adjoint_exact.pre - sol["adjoint"].pre,
            ocp.area, ocp.area_kron, ocp.dt, ocp.Mat)

        if _type == "space":
            error[i, 5] = ocp.mesh.size()
        elif _type == "time":
            error[i, 5] = ocp.dt

        # experimental order of convergence
        if i > 0:
            order[i-1, :] = np.log(error[i-1, :5] / error[i, :5]) \
                / np.log(error[i-1, 5] / error[i, 5])
        i += 1
        eig_max += [ocp.Mat.EIGMAX]
        eig_min += [ocp.Mat.EIGMIN]
        cond_no += [ocp.Mat.CONDNO]

    print_line()
    print(strcolor.cyan_b("\t\t\t\nHFEM: ERRORS\n"))
    print(strcolor.cyan_b(txt))
    for i in range(len(data)):
        print(row_data.format(
            error[i, 0], error[i, 1], error[i, 2], error[i, 3], error[i, 4]))
    print()

    print_line()
    print(strcolor.cyan_b("\t\t\t\nHFEM: ORDER OF CONVERGENCE\n"))
    print(strcolor.cyan_b(txt))
    for i in range(len(data)-1):
        print(row_data.format(
            order[i, 0], order[i, 1], order[i, 2], order[i, 3], order[i, 4]))
    print()

    print_line()
    print(strcolor.cyan_b("\t\t\t\nHFEM: SPECTRAL INFO\n"))
    print(strcolor.cyan_b("Meshsize\tEig Max\t\tEig Min\t\tCond No"))
    for i in range(len(data)):
        print("{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}".format(error[i, 5], eig_max[i],
            eig_min[i], cond_no[i]))
    print()
    print_line()


if __name__ == "__main__":
    print_start()
    print(strcolor.purple_b(
        "\n> HYBRID FINITE ELEMENT: SPATIAL DISCRETIZATION ERRORS \n"))
    hybrid_error_table("space")
    print(strcolor.purple_b(
        "\n> HYBRID FINITE ELEMENT: TEMPORAL DISCRETIZATION ERRORS \n"))
    hybrid_error_table("time")
    print_end()
