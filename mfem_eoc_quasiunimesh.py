"""
Computes the experimental orders of convergence in the mixed method with an initial
quasi-uniform mesh and successively refined by bisection.
"""

from pgrtwave import *
import os

def mixed_error_table():
    """
    Prints the table of errors and orders of convergence for the optimal controls,
    states, and adjoint states in the mixed formulation.

    Spatial errors if <_type="space"> and temporal errors if <_type="time">.
    """

    txt = "Control\t\tVelocity\tPressure\tAdj Velocity\tAdj Pressure"
    row_data = "{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.6e}"

    pwd = os.getcwd() + "/mesh/"
    node_filenames = [pwd + "node0.npy", pwd + "node1.npy",
        pwd + "node2.npy", pwd + "node3.npy", pwd + "node4.npy"]
    tri_filenames = [pwd + "tri0.npy", pwd + "tri1.npy",
        pwd + "tri2.npy", pwd + "tri3.npy", pwd + "tri4.npy"]

    error = np.zeros((len(node_filenames), 6)).astype(float)
    order = np.zeros((len(node_filenames)-1, 5)).astype(float)

    i = 0
    for num in range(len(node_filenames)):
        ocp = OCP(prm=dict(n=10, T=1.0, num_time_step=1000, hybrid=False,
            alpha=1.0, beta=1.0, gamma=1.0, ocptol=1e-6, pressure_ave=False,
            velocity_ave=False, lagrange_ave=False),
            mesh=Mesh(np.load(node_filenames[num]),
            np.load(tri_filenames[num])))
        print_details(ocp.mesh, ocp.Mat, ocp.prm, ocp.dt)

        # x and y coordinates of the nodes in the mesh
        x = ocp.mesh.node[ocp.mesh.tri-1, 0].reshape(3*ocp.mesh.num_elems,)
        y = ocp.mesh.node[ocp.mesh.tri-1, 1].reshape(3*ocp.mesh.num_elems,)

        # construct exact optimal control, state and adjoint state
        adjoint_exact \
            = Wave_Vars(exact_dual_p(ocp.t, x, y),
            exact_dual_v(ocp.t, ocp.mesh))
        state_exact \
            = Wave_Vars(desired_p(ocp.t, x, y), desired_v(ocp.t, ocp.mesh))
        control_exact = - adjoint_exact.pre / ocp.prm["gamma"]

        # construct right hand side and desired state
        ocp.rhs = exact_source(ocp.t, x, y) - control_exact
        ocp.desired \
            = Wave_Vars(state_exact.pre - exact_dual_source(ocp.t, x, y)
            / ocp.prm["alpha"], state_exact.vel)

        # solve optimal control problem
        sol = Barzilai_Borwein(ocp, SecondPoint=None, info=True, version=3)

        # error in control
        error[i, 0] = pre_norm_P1(control_exact -
            np.kron([1, 1, 1], sol["control"]).reshape(3*ocp.mesh.num_elems,
            ocp.prm["num_time_step"]+1), ocp.area, ocp.area_kron,
            ocp.dt, ocp.Mat)
        # error in velocity
        error[i, 1] = vel_norm(state_exact.vel - sol["state"].vel,
            ocp.dt, ocp.Mat)
        # error in pressure
        error[i, 2] = pre_norm_P1(state_exact.pre -
            np.kron([1, 1, 1], sol["state"].pre).reshape(3*ocp.mesh.num_elems,
            ocp.prm["num_time_step"]+1), ocp.area, ocp.area_kron,
            ocp.dt, ocp.Mat)
        # error in dual velocity
        error[i, 3] = vel_norm(adjoint_exact.vel - sol["adjoint"].vel,
            ocp.dt, ocp.Mat)
        # error in dual pressure
        error[i, 4] = pre_norm_P1(adjoint_exact.pre -
            np.kron([1, 1, 1], sol["adjoint"].pre).reshape(3*ocp.mesh.num_elems,
            ocp.prm["num_time_step"]+1), ocp.area, ocp.area_kron,
            ocp.dt, ocp.Mat)

        error[i, 5] = ocp.mesh.size()

        if i > 0:
            order[i-1, :] = np.log(error[i-1, :5] / error[i, :5]) \
                / np.log(error[i-1, 5] / error[i, 5])
        i += 1

    print_line()
    print(strcolor.cyan_b("\t\t\t\nMFEM: ERRORS\n"))
    print(strcolor.cyan_b(txt))
    for i in range(len(node_filenames)):
        print(row_data.format(
            error[i, 0], error[i, 1], error[i, 2], error[i, 3], error[i, 4]))
    print()

    print_line()
    print(strcolor.cyan_b("\t\t\t\nMFEM: ORDER OF CONVERGENCE\n"))
    print(strcolor.cyan_b(txt))
    for i in range(len(node_filenames)-1):
        print(row_data.format(
            order[i, 0], order[i, 1], order[i, 2], order[i, 3], order[i, 4]))
    print()
    print_line()


if __name__ == "__main__":
    print_start()
    print(strcolor.purple_b(
        "\n> MIXED FINITE ELEMENT: SPATIAL DISCRETIZATION ERRORS (QUASIUNIMESH) \n"))
    mixed_error_table()
    print_end()
