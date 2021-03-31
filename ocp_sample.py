"""
An example of an optimal control problem for the wave equation using the PGRT FEM.
(Example 1)
"""

from pgrtwave import *

def plot_surface(ax, x, y, z, mesh, lo, hi, view):
    """
    Plot surface based on a given triangulation.

    Arguments
    ---------
        - ax        plot axis
        - x         x coordinates of nodes
        - y         y coordinates of nodes
        - z         function value at nodes
        - mesh      triangulation of the domain
        - lo        lower limit for z
        - hi        upper limit for z
        - view      elevation view parameter
    """

    triang = tri.Triangulation(x, y, triangles=mesh.tri-1)
    ax.plot_trisurf(triang, z, linewidth=0.2, antialiased=True,
        cmap=cm.Blues, vmin=lo, vmax=hi, shade=True, edgecolor="grey")
    ax.view_init(elev=view)


def plot_per_elem(ax, x, y, z, lo, hi, view):
    """
    Plot the surface per element. See the function <plot_surface> for the meaning
    of the arguemnts.
    """

    tri = np.array([1, 2, 3]) / (2 ** 8)
    ax.plot_trisurf(x, y, z, tri, linewidth=0.2, antialiased=True,
        cmap=cm.Blues, vmin=lo, vmax=hi, shade=True, edgecolor="grey")
    ax.view_init(elev=view)
    ax.axis("tight")


def plot_hybrid(mesh, state, control, Mat, tmesh, t, figfilename, fignum=[1, 2, 3]):
    """
    Plots state and control variables generated from the hybrid FEM.

    Arguments
    ---------
        - mesh          the triangulation of the domain
        - state         state variable
        - control       control variable
        - Mat           matrices in the hybrid FEM
        - tmesh         temporal mesh
        - t             index in temporal mesh to plot
        - showplot      boolean variable for showing plot
    """

    pre_postproc = HFEM_PostProcess(state, Mat, mesh)

    fig1 = plt.figure(fignum[0], figsize=(6, 9))
    ax11 = fig1.add_subplot(321, projection="3d")
    ax12 = fig1.add_subplot(322, projection="3d")
    ax13 = fig1.add_subplot(323, projection="3d")
    ax14 = fig1.add_subplot(324, projection="3d")

    fig2 = plt.figure(fignum[1], figsize=(6, 6))
    ax21 = fig2.add_subplot(221, projection="3d")
    ax22 = fig2.add_subplot(222, projection="3d")
    ax23 = fig2.add_subplot(223, projection="3d")
    ax24 = fig2.add_subplot(224, projection="3d")

    fig3 = plt.figure(fignum[2])
    ax31 = fig3.add_subplot(122, projection="3d")

    num_elems = mesh.num_elems
    x = mesh.node[mesh.tri-1, 0].reshape(3*num_elems,)
    y = mesh.node[mesh.tri-1, 1].reshape(3*num_elems,)

    vel = flux_convert_hybrid(mesh, state.vel[:, t])

    pre_max = np.max(state.pre[:, t])
    pre_min = np.min(state.pre[:, t])
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    ctr_max = np.max(control)
    ctr_min = np.min(control)

    for i in range(num_elems):
        ind = range(3*i, 3*i+3)
        plot_per_elem(ax11, x[ind], y[ind], pre_postproc[ind, t],
            lo=pre_min, hi=pre_max, view=55)
        plot_per_elem(ax12, x[ind], y[ind], [state.pre[i, t]]*3,
            lo=pre_min, hi=pre_max, view=55)
        plot_per_elem(ax21, x[ind], y[ind], vel[ind, 0],
            lo=vel_min, hi=vel_max, view=55)
        plot_per_elem(ax22, x[ind], y[ind], vel[ind, 1],
            lo=vel_min, hi=vel_max, view=55)
        plot_per_elem(ax31, x[ind], y[ind], control[ind, t],
            lo=ctr_min, hi=ctr_max, view=55)

    x = mesh.node[:, 0]
    y = mesh.node[:, 1]
    pre = time_coeff_p(tmesh[t]) * init_p(x, y)
    vt = time_coeff_v(tmesh[t])
    velx, vely =  grad_p(x, y)
    velx, vely = vt * velx, vt * vely

    plot_surface(ax13, x, y, pre, mesh, lo=pre_min, hi=pre_max, view=55)
    plot_surface(ax23, x, y, velx, mesh, lo=vel_min, hi=vel_max, view=55)
    plot_surface(ax24, x, y, vely, mesh, lo=vel_min, hi=vel_max, view=55)

    x = mesh.node[mesh.edge[mesh.int_edge, 0]-1, :]
    y = mesh.node[mesh.edge[mesh.int_edge, 1]-1, :]

    colors = plt.cm.Blues(np.linspace(0, 1, len(mesh.int_edge)))
    state_lag_sort = np.sort(state.lag[:, t])
    state_lag_idx = np.argsort(state.lag[:, t])
    for i in range(len(mesh.int_edge)):
        ax14.plot([x[state_lag_idx[i], 0], y[state_lag_idx[i], 0]],
            [x[state_lag_idx[i], 1], y[state_lag_idx[i], 1]],
            [state_lag_sort[i]] * 2, color=colors[i], lw=1)
    ax14.view_init(elev=55)

    for ax in [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24]:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

    for ax in [ax11, ax12, ax13, ax14]:
        ax.set_zticks([-1, 0, 1])

    for ax in [ax21, ax22, ax23, ax24]:
        ax.set_zticks([-2, 0, 2])

    ax11.set_xlabel("(a)", fontsize="10")
    ax12.set_xlabel("(b)", fontsize="10")
    ax13.set_xlabel("(c)", fontsize="10")
    ax14.set_xlabel("(d)", fontsize="10")
    ax21.set_xlabel("(a)", fontsize="10")
    ax22.set_xlabel("(b)", fontsize="10")
    ax23.set_xlabel("(c)", fontsize="10")
    ax24.set_xlabel("(d)", fontsize="10")

    plt.tight_layout()
    fig2.savefig(figfilename, dpi=1200)
    print(strcolor.highlight("> File created: " + figfilename) + "\n")

    return fig1


def plot_control(mesh, control1, control2, tmesh, fig1, t=-1, figfilename=""):
    """
    Plot the controls in the mixed and hybrid PGRT-FEM.
    """

    ax25 = fig1.add_subplot(325, projection="3d")
    ax26 = fig1.add_subplot(326, projection="3d")
    ctr1_max = np.max(control1[:, t])
    ctr1_min = np.min(control1[:, t])
    ctr2_max = np.max(control2[:, t])
    ctr2_min = np.min(control2[:, t])
    ctr_max = max(ctr1_max, ctr2_max)
    ctr_min = min(ctr1_min, ctr2_min)

    num_elems = mesh.num_elems
    x = mesh.node[mesh.tri-1, 0].reshape(3*num_elems,)
    y = mesh.node[mesh.tri-1, 1].reshape(3*num_elems,)

    for i in range(num_elems):
        ind = range(3*i, 3*i+3)
        plot_per_elem(ax25, x[ind], y[ind], [control1[i, t]]*3,
            lo=ctr_min, hi=ctr_max, view=55)
        plot_per_elem(ax26, x[ind], y[ind], control2[ind, t],
            lo=ctr_min, hi=ctr_max, view=55)

    for ax in [ax25, ax26]:
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_zticks([10, 0, -10])

    ax25.set_xlabel("(e)", fontsize="10")
    ax26.set_xlabel("(f)", fontsize="10")
    plt.tight_layout()

    fig1.savefig(figfilename, dpi=1200)
    print(strcolor.highlight("> File created: " + figfilename) + "\n")


def plot_result_mfem(sol, ocp, figfilename, fignum=4):
    """
    Plot the results for the MFEM.
    """
    fig = plt.figure(fignum, figsize=(12, 3.5))

    ax1 = fig.add_subplot(141)
    ax1.semilogy(range(1, len(sol["costvalues"])+1), sol["costvalues"],
        color="darkblue", label=r"$j_{hk}(u_{hk}^i)$", lw=1.25)
    ax1.legend(loc="best", fontsize="10")
    ax1.autoscale(enable=True, axis="x", tight=True)
    ax1.set_xlabel("(a)", fontsize="10")
    ax1.set_ylim([1e-3, 1e2])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    ax2 = fig.add_subplot(142)
    ax2.semilogy(range(1, len(sol["optresidue"])+1), sol["optresidue"],
        color="darkblue", label=r"$\|\gamma u_{hk}^i + w_{hk}^i\|_I$", lw=1.25)
    ax2.legend(loc="best", fontsize="10")
    ax2.autoscale(enable=True, axis="x", tight=True)
    ax2.set_xlabel("(b)", fontsize="10")
    ax2.set_ylim([1.75e-5, 1e1])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    ax3 = fig.add_subplot(143)
    residual_pre = np.zeros(ocp.prm["num_time_step"])
    residual_vel = np.zeros(ocp.prm["num_time_step"])
    for i in range(ocp.prm["num_time_step"]):
        residual_pre[i] = \
            np.dot(sol["residue"].pre[:, i],
            ocp.area_kron * sol["residue"].pre[:, i])
        residual_vel[i] = np.dot(sol["residue"].vel[:, i],
            ocp.Mat.mss * sol["residue"].vel[:, i])
    ax3.semilogy(ocp.t[2:], residual_pre[1:], color="darkblue",
        label=r"$\|\bar{p}_{hk}(t) - p_{dh}(t)\|_{L^2}$", lw=1.25)
    ax3.semilogy(ocp.t[2:], residual_vel[1:], color="firebrick",
        label=r"$\|\bar{\mathbf{v}}_{hk}(t) - \mathbf{v}_{dh}(t)\|_{L^2}$",
        ls="dashdot", lw=1.25)
    ax3.legend(loc="best", fontsize="10")
    ax3.autoscale(enable=True, axis="x", tight=True)
    ax3.set_xlabel("(c)", fontsize="10")
    ax3.set_ylim([1e-6, 1e-1])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    ax4 = fig.add_subplot(144)
    control_norm = np.zeros(ocp.prm["num_time_step"])
    for i in range(ocp.prm["num_time_step"]):
        control_norm[i] = \
            np.dot(sol["control"][:, i], ocp.area * sol["control"][:, i])
    ax4.plot(ocp.t[1:], control_norm, color="darkblue",
        label=r"$\|\bar{u}_{hk}(t)\|_{L^2}$", lw=1.25)
    ax4.legend(loc="best", fontsize="10")
    ax4.autoscale(enable=True, axis="x", tight=True)
    ax4.set_xlabel("(d)", fontsize="10")
    ax4.set_ylim([50, 600])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    plt.tight_layout()
    fig.savefig(figfilename)

    print(strcolor.highlight("> File created: " + figfilename) + "\n")


def plot_result_hfem(sol, ocp, figfilename, fignum=5):
    """
    Plot the results for the HFEM.
    """
    fig = plt.figure(fignum, figsize=(12, 3.5))

    ax1 = fig.add_subplot(141)
    ax1.semilogy(range(1, len(sol["costvalues"])+1), sol["costvalues"],
        color="darkblue", label=r"$j_{H,hk}(u_{hk}^i)$", lw=1.25)
    ax1.legend(loc="best", fontsize="10")
    ax1.autoscale(enable=True, axis="x", tight=True)
    ax1.set_xlabel("(a)", fontsize="10")
    ax1.set_ylim([1e-3, 1e2])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    ax2 = fig.add_subplot(142)
    ax2.semilogy(range(1, len(sol["optresidue"])+1), sol["optresidue"],
        color="darkblue",
        label=r"$\|\gamma u_{hk}^i + \tilde{\Pi}_{hk}R_h^1\mu_{hk}^i\|_I$",
        lw=1.25)
    ax2.legend(loc="best", fontsize="10")
    ax2.autoscale(enable=True, axis="x", tight=True)
    ax2.set_xlabel("(b)", fontsize="10")
    ax2.set_ylim([1.75e-5, 1e1])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    ax3 = fig.add_subplot(143)
    residual_pre = np.zeros(ocp.prm["num_time_step"])
    residual_vel = np.zeros(ocp.prm["num_time_step"])

    for i in range(ocp.prm["num_time_step"]):
        residual_pre[i] = \
            np.dot(sol["residue"].pre[:, i],
            ocp.Mat.P1 * ocp.area_kron * sol["residue"].pre[:, i])
        residual_vel[i] = np.dot(sol["residue"].vel[:, i],
            ocp.Mat.mss * sol["residue"].vel[:, i])
    ax3.semilogy(ocp.t[2:], residual_pre[1:], color="darkblue",
        label=r"$\|R_h^1\bar{\lambda}_{hk}(t) - p_{dh}(t)\|_{L^2}$", lw=1.25)
    ax3.semilogy(ocp.t[2:], residual_vel[1:], color="firebrick",
        label=r"$\|\bar{\mathbf{v}}_{hk}(t) - \mathbf{v}_{dh}(t)\|_{L^2}$",
        ls="dashdot", lw=1.25)
    ax3.legend(loc="best", fontsize="10")
    ax3.autoscale(enable=True, axis="x", tight=True)
    ax3.set_xlabel("(c)", fontsize="10")
    ax3.set_ylim([1e-6, 1e-1])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    ax4 = fig.add_subplot(144)
    control_norm = np.zeros(ocp.prm["num_time_step"])
    for i in range(ocp.prm["num_time_step"]):
        control_norm[i] = \
            np.dot(sol["control"][:, i],
            ocp.Mat.P1 * ocp.area_kron * sol["control"][:, i])
    ax4.plot(ocp.t[1:], control_norm, color="darkblue",
        label=r"$\|\bar{u}_{hk}(t)\|_{L^2}$", lw=1.25)
    ax4.legend(loc="best", fontsize="10")
    ax4.autoscale(enable=True, axis="x", tight=True)
    ax4.set_xlabel("(d)", fontsize="10")
    ax4.set_ylim([50, 600])
    plt.grid(True, which="major", ls="dotted", color="0.65")

    plt.tight_layout()
    fig.savefig(figfilename)

    print(strcolor.highlight("> File created: " + figfilename) + "\n")


def ocp_example(_type, num=21, time_instant=-1, pressure_ave=False,
    velocity_ave=False, lagrange_ave=False, kwargs=None, result_figfilename="",
    result_hybrid_fignum=1, plot_hybrid_filename=None):
    """
    An example of optimal control problem.

    Arguments
    ---------
        - _type             either "hybrid" or "mixed"
        - num               number of subintervals in an axis
        - time_instant      time instant to plot
        - showplot          boolean for showing plots
        - pressure_ave      boolean for taking the pressure temporal average
        - velocity_ave      boolean for taking the velocity temporal average
        - lagrange_ave      boolean for taking the lagrange temporal average
        - kwargs            are other keyword arguments to be utilized
    """

    if _type == "mixed":
        ocp = OCP(prm=dict(n=num, T=1.0, num_time_step=100, hybrid=False,
            alpha=10.0, beta=1.0, gamma=1e-5, ocptol=1e-6, pressure_ave=pressure_ave,
            velocity_ave=velocity_ave, lagrange_ave=lagrange_ave))
    elif _type == "hybrid":
        ocp = OCP(prm=dict(n=num, T=1.0, num_time_step=100, hybrid=True,
            alpha=10.0, beta=1.0, gamma=1e-5, ocptol=1e-6, pressure_ave=pressure_ave,
            velocity_ave=velocity_ave, lagrange_ave=lagrange_ave))
    if kwargs is not None:
        ocp.prm.update(kwargs)

    print_details(ocp.mesh, ocp.Mat, ocp.prm, ocp.dt)
    sol = Barzilai_Borwein(ocp, SecondPoint=None, version=3)

    fig2 = None
    if _type == "mixed":
        pass
    elif _type == "hybrid" and plot_hybrid_filename is not None:
        fig2 = plot_hybrid(ocp.mesh, sol["state"], sol["control"], ocp.Mat,
            tmesh=ocp.t, t=time_instant, figfilename=plot_hybrid_filename)

    if _type == "mixed":
        plot_result_mfem(sol, ocp, figfilename=result_figfilename)
    elif _type == "hybrid":
        plot_result_hfem(sol, ocp, figfilename=result_figfilename,
            fignum=result_hybrid_fignum)

    return sol["control"], ocp.mesh, ocp.t, fig2


if __name__ == "__main__":
    print_start()
    control1, _, _, _ = ocp_example("mixed", pressure_ave=False,
        velocity_ave=False, lagrange_ave=False,
        result_figfilename="result_mfem.pdf")
    control2, _, _, _ = ocp_example("hybrid", pressure_ave=False,
        velocity_ave=False, lagrange_ave=False,
        result_figfilename="result_hfem.pdf", result_hybrid_fignum=5)
    control2, mesh, tmesh, fig1 = ocp_example("hybrid", pressure_ave=True,
        velocity_ave=False, lagrange_ave=True,
        result_figfilename="result_hfem_ave.pdf", result_hybrid_fignum=6,
        plot_hybrid_filename="velocity.png")
    plot_control(mesh, control1, control2, tmesh, fig1,
        figfilename="pressure_control.png")
    print_end()
