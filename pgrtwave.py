"""
==========================================================================================
                MFEM AND HFEM FOR OPTIMAL CONTROL OF WAVE EQUATION
==========================================================================================

This Python module approximates a distributed optimal control for the wave equation
written in pressure-velocity formulation using the Petrov-Galerkin-Raviart-Thomas (PGRT)
FEM. For the details, refer to the paper:

G. Peralta and K. Kunisch, Mixed and hybrid Petrov-Galerkin finite element discretization
for optimal control of the wave equation, preprint.

Gilbert Peralta
Department of Mathematics and Computer Science
University of the Philippines Baguio
Governor Pack Road, Baguio, Philippines 2600
Email: grperalta@up.edu.ph
"""

from __future__ import division
from numpy import linalg as la
from scipy import sparse as sp
from scipy.sparse.linalg import eigs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib import tri
import numpy as np
import warnings
import datetime
import platform
import strcolor
import psutil
import time
import sys


__author__ = "Gilbert Peralta"
__copyright__ = "Copyright 2021, Gilbert Peralta"
__version__ = "1.0"
__maintainer__ = "The author"
__institution__ = "University of the Philippines Baguio"
__email__ = "grperalta@up.edu.ph"
__date__ = "15 March 2021"

# LaTex font
rc("font", **{"family": "DejaVu Sans", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

# constant pi
PI = np.math.pi

def mesh_repr(obj):
    """
    Returns the string representation of <Mesh> class.
    """

    np.set_printoptions(threshold=np.inf)
    string = ""

    for key, val in obj.__dict__.items():
        txt_attr = "Attribute: {}\n"
        txt_dscr = "Description: {}"
        txt_remk = "Remark: {}"
        if type(val) in [sp.csr.csr_matrix, sp.csc.csc_matrix,
                         sp.bsr.bsr_matrix]:
            val = val.tocoo()
            string +=  txt_attr.format(str(key))
            if key == "node_to_edge":
                string += txt_dscr.format("Node to edge data structure.")
                string += "\n" + txt_remk.format("Indeing starts at 1.") + "\n"
            for row, col, value in zip(val.row, val.col, val.data):
                string += "({}, {}) {}".format(row, col, value) + "\n"
            string += "\n"
        else:
            string += txt_attr.format(str(key))
            if key == "node":
                string += txt_dscr.format("Coordinates of the nodes.")
            elif key == "tri":
                string += txt_dscr.format("Node connectivity"
                    + " defining the triangles.")
                string += "\n" + txt_remk.format("Indexing starts at 1.")
            elif key == "num_elems":
                string += txt_dscr.format("Number of elements.")
            elif key == "num_edges":
                string += txt_dscr.format("Number of edges.")
            elif key == "edge":
                string += txt_dscr.format("Node connectivity"
                    + " defining the edges.")
            elif key == "elem_center":
                string += txt_dscr.format("Coordinates of the centers"
                    + " of the triangles.")
            elif key == "edge_to_elem":
                string += txt_dscr.format("Edge to element data structure.")
            elif key == "elem_to_edge":
                string += txt_dscr.format("Element to edge data structure.")
                string += "\n" + txt_remk.format("Indexing starts at 1.")
            elif key == "elem_to_edge_sgn":
                string += txt_dscr.format("Global orientation of the edges"
                    + " in the triangulation.")
            elif key == "all_edge":
                string += txt_dscr.format("Local to global index map"
                    + " for the dof associated to the edges.")
                string += "\n" + txt_remk.format("Indexing starts at 1.")
            elif key == "int_edge":
                string += txt_dscr.format("Index of interior nodes.")
                string += "\n" + txt_remk.format("Indexing starts at 1.")
            elif key == "num_nodes":
                string += txt_dscr.format("Number of nodes.")
            string += "\n" + str(val) + "\n\n"

    return string


class Mesh:
    """
    The mesh class for the triangulation of the domain in the finite element method.
    """

    def __init__(self, node, tri):
        """
        Class initialization/construction.

        Keyword arguments
        -----------------
            - node              array of coordinates of the nodes in the mesh
            - tri               array of geometric connectivity of the
                                elements/triangles with respect to the
                                global indices in the array <node>

        Attributes
        ----------
            - node              array of nodes (see key argument node)
            - tri               array of triangles (see key argument tri)
            - edge              array of edges with respect to the ordering
                                of the indices in <node>
            - num_nodes         number of nodes
            - num_elems         number of triangles
            - num_edges         number of edges
            - all_edge          array of all edges counting multiplicity for
                                each elements
            - int_edge          array of interior edges with respect to the
                                ordering in <edge>
            - elem_center       array of barycentes of the elements in the
                                mesh
            - node_to_edge      The node to edge data structure. The matrix
                                with entries such that <node_to_edge(k, l)=j>
                                if the jth edge belongs to the kth and lth
                                nodes and <node_to_edge(k, l)=0> otherwise.
            - edge_to_elem      The edge to element data structure. The matrix
                                such that the jth row is [k, l, m, n] where
                                k is the initial node, l is the terminal node,
                                and m and n are the indices of the elements
                                sharing a common edge, where n = 0 if there
                                is only one triangle containing the edge.
            - elem_to_edge      The element to edge data structure. The matrix
                                such that the jth row is [k, l, m] where k,
                                l, and m are the indices of the edge of the
                                jth element.
            - elem_to_edge_sgn  The sign of the edges with respect to a global
                                fixed orientation.
        """

        self.node = node
        self.tri = tri
        self.num_elems = self.tri.shape[0]
        self.num_nodes = self.node.shape[0]

        # get mesh data structure
        meshinfo = get_mesh_info(node, tri)

        self.edge = meshinfo["edge"]
        self.num_edges = meshinfo["num_edges"]
        self.elem_center = meshinfo["elem_center"]
        self.node_to_edge = meshinfo["node_to_edge"]
        self.edge_to_elem = meshinfo["edge_to_elem"]
        self.elem_to_edge = meshinfo["elem_to_edge"]
        self.elem_to_edge_sgn = meshinfo["elem_to_edge_sgn"]
        self.all_edge = self.elem_to_edge.reshape(3*self.num_elems,)
        self.int_edge = sp.find(self.edge_to_elem[:, 3] != 0)[1]

    def __repr__(self):
        """
        Class string representation.
        """

        txt = "="*78 + "\n"
        txt += "\t\t\tMESH DATA STRUCTURE\n" + "="*78 + "\n\n"
        txt += mesh_repr(self)  + "="*78
        return txt

    def size(self):
        """
        Returns the smallest interior angles (in degrees) of the triangles.
        """

        h = 0.0
        for elem in range(self.num_elems):
            edge1 = (self.node[self.tri[elem, 1]-1, :]
                - self.node[self.tri[elem, 0]-1, :])
            edge2 = (self.node[self.tri[elem, 2]-1, :]
                - self.node[self.tri[elem, 1]-1, :])
            edge3 = (self.node[self.tri[elem, 0]-1, :]
                - self.node[self.tri[elem, 2]-1, :])
            h = max(h, la.norm(edge1), la.norm(edge2), la.norm(edge3))

        return h

    def plot(self, **kwargs):
        """
        Plots the mesh.
        """

        plt.figure()
        plt.triplot(self.node[:, 0], self.node[:, 1],
                    self.tri - 1, "b-", lw=1.0, **kwargs)
        plt.show()


def square_uni_trimesh(n):
    """
    Generates a uniform triangular mesh of the unit square.

    Keyword argument
    ----------------
        - n         number of nodes on one side of the unit square

    Return
    ------
        A <Mesh> class of the uniform triangulation.
    """

    # number of elements
    num_elems = 2 * (n - 1) ** 2

    # pre-allocation of node list
    node = np.zeros((n**2, 2), dtype=np.float)

    # generation of node list
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            # node index
            index = (i - 1)*n + j - 1
            # x-coordinate of a node
            node[index, 0] = float((j-1)) / (n-1)
            # y-coordinate of a node
            node[index, 1] = float((i-1)) / (n-1)

    # pre-allocation of triangular elements
    tri = np.zeros((num_elems, 3), dtype=np.int)
    ctr = 0;

    # generation of triangular elements
    for i in range(1, n):
        for j in range(1, n):
            # lower right node of the square determined by two intersecting
            # triangles
            lr_node = (i-1) * n + j + 1
            # lower left triangle
            tri[ctr, :] = [lr_node, lr_node + n, lr_node - 1]
            # upper right triangle
            tri[ctr+1, :] = [lr_node + n - 1, lr_node - 1, lr_node + n]
            # increment counter
            ctr = ctr + 2

    return Mesh(node, tri)


def get_mesh_info(node, tri):
    """
    Returns the mesh data structure, to be called by the <Mesh> class.

    Keyword arguments
    ------------------
        - node      array of coordinates of the nodes
        - tri       array of triangles

    Reference
    ----------
        C. Bahriawati and C. Carstensen, Three Matlab implementation of the lowest-order
        Raviart-Thomas MFEM with a posteriori error control, Comp. Meth. App. Math. 5,
        pp. 333-361, 2005.
    """

    num_nodes = node.shape[0]
    num_elems = tri.shape[0]

    # pre-allocation of and column indices, and entries for node_to_elem
    row = np.zeros((3*num_elems,), dtype=np.int)
    col = np.zeros((3*num_elems,), dtype=np.int)
    ent = np.zeros((3*num_elems,), dtype=np.int)

    # generation row and column indices, and entries for node to element
    # data structure
    for i in range(num_elems):
        row[3*i : 3*i+3] = tri[i, :]-1
        col[3*i : 3*i+3] = tri[i, [1, 2, 0]]-1
        ent[3*i : 3*i+3] = [i+1, i+1, i+1]

    # node to element data stucture
    node_to_elem = sp.coo_matrix((ent, (row, col)),
        shape=(num_nodes, num_nodes), dtype=int).tocsr()

    # nonzero entries in the lower triangular part of node_to_elem
    nz = sp.find(sp.tril(node_to_elem + node_to_elem.T))

    # number of edges
    num_edges = len(nz[1])

    # node to edge data data structure
    node_to_edge = sp.coo_matrix((range(1, num_edges+1), (nz[1], nz[0])),
        shape=(num_nodes, num_nodes), dtype=np.int).tocsr()
    node_to_edge = node_to_edge + node_to_edge.T

    # pre-allocation of edge to element data stucture
    edge_to_elem = np.zeros((num_edges, 4), dtype=np.int)

    # assembly of edge to element data structure
    for i in range(num_elems):
        for j in range(3):
            initial_node = tri[i, j]
            terminal_node = tri[i, (j%3 + 1) % 3]
            index = node_to_edge[initial_node-1, terminal_node-1]
            if edge_to_elem[index-1, 0] == 0:
                edge_to_elem[index-1, :] = \
                    [initial_node, terminal_node,
                     node_to_elem[initial_node-1, terminal_node-1],
                     node_to_elem[terminal_node-1, initial_node-1]]

    # edges of the mesh
    edge = edge_to_elem[:, 0 : 2]

    # element to edge data stucture
    elem_to_edge = np.zeros((num_elems, 3), dtype=np.int)
    elem_to_edge[:, 0] = node_to_edge[tri[:, 1]-1, tri[:, 2]-1]
    elem_to_edge[:, 1] = node_to_edge[tri[:, 2]-1, tri[:, 0]-1]
    elem_to_edge[:, 2] = node_to_edge[tri[:, 0]-1, tri[:, 1]-1]

    # signs for the element to edge
    elem_to_edge_sgn = np.ones((num_elems, 3), dtype=np.int)

    # pre-allocation centers of each elements
    elem_center = np.zeros((num_elems, 2), dtype=np.float)

    # generates the barycenters of the triangles
    for i in range(num_elems):
        find_index = sp.find(i + 1 == edge_to_elem[elem_to_edge[i, :]-1, 3])
        elem_to_edge_sgn[i, find_index[1]] = - 1
        elem_center[i, 0] = np.sum(node[tri[i, :]-1, 0]) / 3
        elem_center[i, 1] = np.sum(node[tri[i, :]-1, 1]) / 3

    return {"edge":edge, "node_to_edge":node_to_edge,
            "num_edges":num_edges, "edge_to_elem":edge_to_elem,
            "elem_to_edge":elem_to_edge, "elem_center":elem_center,
            "elem_to_edge_sgn":elem_to_edge_sgn}


def gauss1D_quad(l, a, b):
    """
    One-dimensional Gauss integration.

    Keyword arguments
    -----------------
        - l     number of quadrature nodes
        - a     left endpoint of the interval of integration
        - b     right endpoint of the interval of integration

    Return
    -------
        A dictionary with the keys <"nodes">, <"weights">, <"order">, <"dim">
        corresponding to the quadrature nodes, quadrature weights, order and dimension
        of the numerical quadrature.
    """

    dim = 1
    m = float((a + b)) / 2
    delta = float(b - a)

    if l == 1:
        nodes = np.array([m])
        weights = np.array([delta])
        order = 1
    elif l == 2:
        const = delta * np.sqrt(3) / 6
        nodes = np.array([m-const, m+const])
        weights = delta * np.array([0.5, 0.5])
        order = 3
    elif l == 3:
        const = delta * np.sqrt(3./5) / 2
        nodes = np.array([m-const, m, m+const])
        weights = delta * np.array([5./18, 8./18, 5./18])
        order = 5
    else:
        nodes = None
        weights = None
        order = None
        dim = None
        print("Gauss quadrature only available up to order 5 only.")

    return {"nodes":nodes, "weights":weights, "order":order, "dim":dim}


def tri_quad(num_quad):
    """
    Gauss integration on the unit triangle with vertices at (0,0), (0,1) and (1,0).

    Keyword argument
    -----------------
        - num_quad      number of quadrature nodes

    Return
    -------
        A dictionary with the keys <"nodes">, <"weights">, <"order">, <"dim">
        corresponding to the quadrature nodes, quadrature weights, order and dimension
        of the numerical quadrature.

    To do
    -----
        * Include quadrature nodes higher than 6.
    """

    dim = 2

    # change the number of quadrature nodes to the next number
    if num_quad == 2:
        num_quad = 3
        print("Number of quadrature nodes changed from 2 to 3.")
    elif num_quad == 5:
        num_quad = 6
        print("Number of quadrature nodes changed from 5 to 6.")

    if num_quad == 1:
        nodes = np.matrix([1./3, 1./3])
        weights = np.matrix([1./2])
        order = 1
    elif num_quad == 3:
        nodes = np.array([[2./3, 1./6],
                          [1./6, 2./3],
                          [1./6, 1./6]])
        weights = (1./2) * np.array([1, 1, 1]) / 3
        order = 2
    elif num_quad == 4:
        nodes = np.array([[1./3, 1./3],
                          [1./5, 1./5],
                          [3./5, 1./5],
                          [1./5, 3./5]])
        weights = (1./2) * np.array([-27, 25, 25, 25]) / 48
        order = 3
    elif num_quad == 6:
        nodes = np.array([[0.816847572980459, 0.091576213509771],
                          [0.091576213509771, 0.816847572980459],
                          [0.091576213509771, 0.091576213509771],
                          [0.108103018168070, 0.445948490915965],
                          [0.445948490915965, 0.108103018168070],
                          [0.445948490915965, 0.445948490915965]])
        weights = np.array([0.109951743655322,
                            0.109951743655322,
                            0.109951743655322,
                            0.223381589678011,
                            0.223381589678011,
                            0.223381589678011]) * (1./2)
        order = 4
    else:
        nodes = None
        weights = None
        order = None
        dim = None
        print("Number of quadrature nodes available up to 6 only.")

    return {"nodes":nodes, "weights":weights, "order":order, "dim":dim}


def RT0_basis(p):
    """
    Generates the function values and divergence of the lowest order Raviart-Thomas
    finite element at the array of points p.

    Keyword argument
    -----------------
        - p     array of points in the two-dimesional space

    Return
    -------
        A dictionary with keys <"val"> and <"div"> corresponding to the function and
        divergence values at p, with shapes (3, N, 1) and (3, N, 1) where N is the
        number of points in p.
    """

    x = p[:, 0]
    y = p[:, 1]
    val = np.zeros((3, p.shape[0], 2), dtype=np.float)
    div = np.zeros((3, p.shape[0], 1), dtype=np.float) + 2

    val[0, :, :] = np.array([x, y]).T
    val[1, :, :] = np.array([x-1, y]).T
    val[2, :, :] = np.array([x, y-1]).T

    return {"val":val, "div":div}


def affine_transform(mesh):
    """
    Generates the transformations from the reference triangle with vertices at
    (0,0), (0,1) and (1,0) to each element of the mesh.

    Keyword argument
    ----------------
        - mesh      the domain triangulation (a class <Mesh>)

    Return
    -------
        A dictionary with keys <"mat">, <"vec">, <"det"> corresponding to A, b,
        and det(A), where Tx = Ax + b is the linear transformation from the reference
        element to an element in the mesh.
    """

    num_elems = mesh.num_elems
    B_K = np.zeros((num_elems, 2, 2), dtype=np.float)

    # coordinates of the triangles with local indices 0, 1, 2, respectively
    A = mesh.node[mesh.tri[:, 0]-1, :]
    B = mesh.node[mesh.tri[:, 1]-1, :]
    C = mesh.node[mesh.tri[:, 2]-1, :]

    a = B - A
    b = C - A
    B_K[:, :, 0] = a
    B_K[:, :, 1] = b
    B_K_det = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]

    return {"mat":B_K, "vec":A, "det":B_K_det}


def RT0_assemble(mesh, transformations, num_quad=6):
    """
    Assembles the mass and stiffness matrices for the lowest order Raviart-Thomas finite
    element using the edge basis formulation.

    Keyword arguments
    -----------------
        - mesh              the domain triangulation (a class <Mesh>)
        - transformations   dictionary of transformations between each
                            triangle in the mesh and the reference triangle
        - num_quad          (optional) number of quadrature nodes with
                            default value 6

    Return
    ------
        Returns a tuple corresponding to the mass and stiffness matrices.

    To do
    -----
        * Faster assembly, e.g. vectorization.
    """

    # number of elements and edges
    num_edges = mesh.num_edges
    num_elems = mesh.num_elems

    # absolute value for the Jacobian of the matrices of transformations
    B_K_detA = np.abs(transformations["det"])

    # set-up quadrature class
    quad = tri_quad(num_quad)

    # get the number of integration points in the numerical quadrature
    num_int_pts	= quad["nodes"].shape[0]

    # compute the values and divergence values of the basis functions
    # at the quadrature nodes
    rt0 = RT0_basis(quad["nodes"])

    # pre-allocation of row and column indices, and entries of the mass
    # matrix
    row = np.zeros((9*num_elems,), dtype=np.int)
    col = np.zeros((9*num_elems,), dtype=np.int)
    ent = np.zeros((9*num_elems,), dtype=np.float)

    for i in range(num_elems):
        # indices of the edges of the ith element
        ind = mesh.elem_to_edge[i, :] - 1
        ctri = 9 * i
        for j in range(3):
            # sign of the jth edge of the ith element
            sgnj = mesh.elem_to_edge_sgn[i, j]
            ctrj = 3 * j
            for k in range(3):
                ctrk = ctri + ctrj + k
                row[ctrk] = ind[j]
                col[ctrk] = ind[k]
                # sign of the kth edge of the ith element
                sgnk = mesh.elem_to_edge_sgn[i, k]
                local_ent = 0
                for m in range(num_int_pts):
                    local_ent = local_ent + quad["weights"][m] \
                        * np.dot(np.dot(transformations["mat"][i, :, :],
                        rt0["val"][j, m, :]),
                        np.dot(transformations["mat"][i, :, :],
                        rt0["val"][k, m, :]))
                    ent[ctrk] = sgnj * sgnk * (1./B_K_detA[i]) * local_ent

    # assembly of the mass matrix
    A = sp.coo_matrix((ent, (row, col)),
        shape=(num_edges, num_edges), dtype=np.float).tocsr()
    A = 0.5 * (A + A.T)

    # pre-allocation of row and column indices, and entries of the stiffness
    # matrix
    row = np.zeros((3*num_elems,), dtype=np.int)
    col = np.zeros((3*num_elems,), dtype=np.int)
    ent = np.zeros((3*num_elems,), dtype=np.float)

    for i in range(num_elems):
        # indices of the edges of the ith element
        ind = mesh.elem_to_edge[i, :] - 1
        ctri = 3 * i
        for j in range(3):
            ctrj = ctri + j
            row[ctrj] = ind[j]
            col[ctrj] = i
            # sign of the jth edge of the ith element
            sgnj = mesh.elem_to_edge_sgn[i, j]
            local_ent = 0
            for m in range(num_int_pts):
                local_ent = local_ent \
                    + np.dot(quad["weights"][m], rt0["div"][j, m, :])
                ent[ctrj] = sgnj * local_ent

    # assembly of the stiffness matrix
    B = sp.coo_matrix((ent, (row, col)),
        shape=(num_edges, num_elems), dtype=np.float).tocsr()

    return A, B


def RT0_hybrid_assemble(mesh, transformations):
    """
    Assembles the mass, stiffness, and Langrane matrices for the lowest order
    Raviart-Thomas finite element using hybridization.

    Keyword argument
    ----------------
        - mesh              the domain triangulation (a class mesh)
        - transformations   dictionary of transformations between each
                            triangle in the mesh and the reference triangle

    Return
    ------
        A tuple corresponding to the mass, stiffness and Lagrange matrices.

    To do
    -----
        * Faster assembly, e.g. vectorization.
    """

    # number of edges and elements
    num_edges = mesh.num_edges
    num_elems = mesh.num_elems

    # absolute value for the Jacobian of the matrices of transformations
    B_K_detA = np.abs(transformations["det"])

    # pre-allocation of the row/column indices and entries of the mass
    # matrix
    ind = np.zeros((3*num_elems,), dtype=np.int)
    ent = np.zeros((3*num_elems,), dtype=np.float)

    # all nodes counting multiplicity in each triangles of the mesh
    pts = mesh.node[mesh.tri[0, :]-1, :]

    # sum of squares of the edge lengths for each triangle
    s = np.linalg.norm(pts[0, :] - pts[1, :])**2 \
        + np.linalg.norm(pts[1, :] - pts[2, :])**2 \
        + np.linalg.norm(pts[2, :] - pts[0, :])**2

    ind = np.array(range(3*num_elems))
    ent = 0.5 * (np.kron(B_K_detA, [1, 1, 1])
                 * np.array([1, 1, s/36] * num_elems))

    # assembly of mass matrix
    A = sp.coo_matrix((ent, (ind, ind)), shape=(3*num_elems, 3*num_elems),
                      dtype=np.float).tocsr()
    A = 0.5 * (A + A.T)

    # row and column indices, and entries of the stiffness matrix
    row = 3 * np.array(range(num_elems), dtype=np.int) + 2
    col = np.array(range(num_elems), dtype=np.int)
    ent = B_K_detA

    # assembly of the stiffness matrix
    B = sp.coo_matrix((ent, (row, col)),
        shape=(3*num_elems, num_elems), dtype=np.float).tocsr()

    # number of interior edges
    num_int_edges = len(mesh.int_edge)

    # pre-allocation of row, column indices, and entries of the
    # matrix associated with the lagrange multiplier
    row = np.zeros((6*num_int_edges,), dtype=np.int)
    col = np.zeros((6*num_int_edges,), dtype=np.int)
    ent = np.zeros((6*num_int_edges,), dtype=np.float)

    for i in range(num_int_edges):
        initial_node = mesh.node[mesh.edge[mesh.int_edge[i], 0]-1, :]
        terminal_node = mesh.node[mesh.edge[mesh.int_edge[i], 1]-1, :]
        edge = terminal_node - initial_node
        elem_plus = mesh.edge_to_elem[mesh.int_edge[i], 2] - 1
        elem_minus = mesh.edge_to_elem[mesh.int_edge[i], 3] - 1
        mp_plus = mesh.elem_center[elem_plus, :]
        mp_minus = mesh.elem_center[elem_minus, :]
        ent_plus = np.matrix([mp_plus[1] - initial_node[1],
            initial_node[0] - mp_plus[0]])*np.matrix(edge).T
        ent_minus = np.matrix([mp_minus[1] - initial_node[1],
            initial_node[0] - mp_minus[0]])*np.matrix(edge).T
        row[6*i : 6*i+3] = range(3*elem_plus, 3*elem_plus+3)
        row[6*i+3: 6*i+6] = range(3*elem_minus, 3*elem_minus+3)
        col[6*i : 6*i+6] = [i] * 6
        ent[6*i : 6*i+6] = [-edge[1], edge[0], -ent_plus,
            edge[1], -edge[0], ent_minus]

    # assembly of the matrix associated with the Lagrange multiplier
    C = sp.coo_matrix((ent, (row, col)),
        shape=(3*num_elems, num_int_edges), dtype = np.float).tocsr()

    return A, B, C


def mat_dg_P0P1(num_elems):
    """
    Mass matrix corresponding to the inner product of discontinous P0 and discontinuous
    P1 Lagrange elements divided by the area of the triangles.

    Keyword argument
    ----------------
        - num_elems         number of elements in the mesh
    """

    return sp.kron(sp.eye(num_elems), [1./3, 1./3, 1./3])


def mat_dg_P1(num_elems):
    """
    Mass matrix corresponding to the discontinuous P1 Lagrange element divided by the
    area of the triangles.

    Keyword argument
    ----------------
        - num_elems         number of elements in the mesh
    """

    local_ent = np.array([[2.0, 1.0, 1.0],
                          [1.0, 2.0, 1.0],
                          [1.0, 1.0, 2.0]]) * (1./12)

    return sp.kron(sp.eye(num_elems), local_ent)


def mat_postproc(num_elems):
    """
    Matrix required in the post-processing the Lagrange multipliers.

    Keyword argument
    ----------------
        - num_elems         number of elements in the mesh
    """

    local_ent = np.array([[-1.0, 1.0, 1.0],
                          [1.0, -1.0, 1.0],
                          [1.0, 1.0, -1.0]])

    return sp.kron(sp.eye(num_elems), local_ent)


def mat_extend(mesh):
    """
    Matrix required in extending the Lagrange multipliers.

    Keyword argument
    ----------------
        - mesh         the domain triangulation (a class <Mesh>)
    """

    row = np.array(range(3*mesh.num_elems), dtype=np.int)
    col = mesh.elem_to_edge-1
    col = col.flatten()

    M = sp.coo_matrix((np.ones(3*mesh.num_elems, dtype=np.float), (row, col)),
        shape=(3*mesh.num_elems, mesh.num_edges), dtype=np.float).tocsc()

    return M


def default_parameters():
    """
    Returns a dictionary for the default parameters of the optimal control problems.
    """

    return dict(pen=1e-10, alpha=1.0, beta=1.0, gamma=1e-6,
                cgmaxit=None, cgtol=1e-12, ocptol=1e-6,
                ocpmaxit=1000, hybrid=False, pressure_ave=True,
                velocity_ave=False, lagrange_ave=True)


class Matrices:
    """
    The class for the matrices in the lowest order Raviart-Thomas FE.
    """

    def __init__(self, mesh, prm, dt, area, area_inv, transformations=None):
        """
        Class initialization/construction.

        Attributes
        ----------
            - mss           mass matrix
            - stf           stiffness matrix
            - total         total matrix for the linear systems
            - lag           Lagrange matrix in the hybrid formulation
            - postproc      post-processing matrix in the hybrid formulation
            - extend        matrix used in extending Lagrange multipliers
            - extint        matrix used in extending the interior Lagrange
                            multipliers
            - P1            mass matrix for the discontinuous P1 elements
            - P0P1          mass matrix for the inner product between
                            P0 and P1 elemebnts
            - M_plus        total matrix for right hand side
            - M_minus       total matrix for left hand side
            - P_inv         inverse of preconditioner

        """

        if transformations is None:
            transformations = affine_transform(mesh)

        if not prm["hybrid"]:
            self.mss, self.stf = RT0_assemble(mesh, transformations)
            self.M_plus = self.mss \
                + 0.25 * (dt**2) * (self.stf * area_inv * self.stf.T)
            self.M_minus = self.mss \
                - 0.25 * (dt**2) * (self.stf * area_inv * self.stf.T)
            self.lag = None
        else:
            self.mss, self.stf, self.lag \
                = RT0_hybrid_assemble(mesh, transformations)
            self.M_plus = self.mss \
                + (dt/2) * (self.lag * self.lag.T) / prm["pen"] \
                + 0.25 * (dt**2) * (self.stf * area_inv * self.stf.T)
            self.M_minus = self.mss \
                - (dt/2) * (self.lag * self.lag.T) / prm["pen"] \
                - 0.25 * (dt**2) * (self.stf * area_inv * self.stf.T)
            P = self.mss \
                + 0.25 * (dt**2) * (self.stf * area_inv * self.stf.T)
            self.P_inv = 1 / P.diagonal()
            self.postproc = mat_postproc(mesh.num_elems)
            self.extend = mat_extend(mesh)
            self.extint = self.extend[:, mesh.int_edge]
        self.P1 = mat_dg_P1(mesh.num_elems)
        self.P0P1 = mat_dg_P0P1(mesh.num_elems)


def sparse_matrix_density(sp_matrix):
    """
    Calculates the density of the sparse matrix sp_matrix, that is, the ratio between
    the number of nonzero entries and the size of the matrix.
    """

    nnz = len(sp.find(sp_matrix)[2])
    return nnz / (sp_matrix.shape[0] * sp_matrix.shape[1])


def plot_sparse_matrix(sp_matrix, fn=1, info=True, ms=1):
    """
    Plot the sparse matrix.

    Keyword arguments
    -----------------
        - sp_matrix     the sparse matrix (type <scipy.sparse.coo_matrix>)
        - fn            figure number window (default 1)
        - info          boolean variable if to print the size and
                        density of the matrix (default <True>)
        - ms            markersize (default 1)
    """

    fig = plt.figure(fn)
    plt.spy(sp_matrix, markersize=ms)
    plt.xticks([])
    plt.yticks([])
    if info == True:
        density = sparse_matrix_density(sp_matrix)
        string = ("Size of Matrix : {}-by-{}\nDensity : {:.4f}")
        plt.xlabel(string.format(sp_matrix.shape[0],
            sp_matrix.shape[1], density))
    plt.show()


def flux_interp(mesh):
    """
    Interpolation of flux in the lowest order Raviart-Thomas FE.

    Returns the coefficients of the edge basis functions for the RT0-FE. The coefficients
    are the integrals of the normal component of the flux along the edges. This line
    integral is approximated using one-dimensional Gaussian quadrature. The default
    number of nodes for the quadrature is 3.

    Keyword Arguments
    -----------------
        - mesh              the domain triangulation

    Returns
    -------
        The global interpolation of the flux provided by the function
        grad_p in functions.py.
    """

    # number of elements and edges
    num_elems = mesh.num_elems
    num_edges = mesh.num_edges

    # pre-allocation of the interpolated flux
    p = np.zeros((num_edges,), dtype=np.float)

    # set-up one dimensional gaussian quadrature
    quad = gauss1D_quad(3, 0, 1)

    for i in range(num_edges):
        # initial and terminal nodes of the edge
        initial_node = mesh.node[mesh.edge[i, 0]-1, :]
        terminal_node = mesh.node[mesh.edge[i, 1]-1, :]
        # vector corresponding to the edge
        edge = terminal_node - initial_node
        edgelength = np.linalg.norm(edge)
        # normal component of the edge
        normal = [edge[1], - edge[0]] / edgelength
        # coordinates of the gauss nodes on the edge
        x = quad["nodes"] * initial_node[0] \
            + (1 - quad["nodes"]) * terminal_node[0]
        y = quad["nodes"] * initial_node[1] \
            + (1 - quad["nodes"]) * terminal_node[1]
        # function values of the flux at the gauss nodes
        (px, py) = grad_p(x, y)
        # coefficient of the interpolated flux on the edge
        p[i] = (normal[0] * np.dot(quad["weights"], px)
            + normal[1] * np.dot(quad["weights"], py)) * edgelength

    return p


def flux_convert(mesh, transformations, p):
    """
    Conversion of flux in terms of the coefficients of local barycentric coordinates.

    Keyword Arguments
    -----------------
        - mesh              the domain triangulation
        - transformations   the affine transformations from the reference
                            triangle to each triangle of the mesh
        - p                 coefficients of the flux as a linear combination
                            of the edge basis functions

    Return
    ------
        Coefficients of the local barycentric coordinates of each triangles.
    """

    # number of elements
    num_elems = mesh.num_elems

    # pre-allocation of coefficient vector
    cnvrt_p = np.zeros((3*num_elems, 2), dtype=np.float)

    for i in range(num_elems):
        # ith element
        elem = mesh.tri[i, :] - 1
        # sign of the edges in the element
        sgn = mesh.elem_to_edge_sgn[i, :]
        # coefficients of p with respect to the edges of the element
        c = p[mesh.elem_to_edge[i, :]-1]
        pt1 = mesh.node[elem[0], :] / transformations["det"][i]
        pt2 = mesh.node[elem[1], :] / transformations["det"][i]
        pt3 = mesh.node[elem[2], :] / transformations["det"][i]
        coeff1 = sgn[1]*c[1]*(pt1 - pt2) + sgn[2]*c[2]*(pt1 - pt3);
        coeff2 = sgn[0]*c[0]*(pt2 - pt1) + sgn[2]*c[2]*(pt2 - pt3);
        coeff3 = sgn[1]*c[1]*(pt3 - pt2) + sgn[0]*c[0]*(pt3 - pt1);
        # x and y coordinates of the coefficient vector
        cnvrt_p[3*i : 3*i+3, 0] = [coeff1[0], coeff2[0], coeff3[0]]
        cnvrt_p[3*i : 3*i+3, 1] = [coeff1[1], coeff2[1], coeff3[1]]

    return cnvrt_p


def flux_interp_hybrid(mesh, p):
    """
    Interpolation for the hybridized lowest Raviart-Thomas finite elements.

    Keyword arguments
    -----------------
        - mesh      the domain triangulation
        - p         coefficients corresponding to the global interpolation
                    of the flux using edge basis functions

    Return
    ------
        Coefficients of the flux in each element corresponding to the
        local basis functions (1,0), (0,1) and (x - x_b, y - y_b) where
        (x_b, y_b) is the barycenter of the triangle.
    """

    # number of elements
    num_elems = mesh.num_elems

    # pre-allocation of the coefficient vector
    interp_p = np.zeros((3*num_elems,), dtype=np.float)

    for i in range(num_elems):
        # ith element
        elem = mesh.tri[i, :] - 1
        # coordinates of the current element
        x_coord = mesh.node[elem, 0]
        y_coord = mesh.node[elem, 1]
        if x_coord[1] == x_coord[0]:
            index = [3*i, 3*i+2]
            coord = [x_coord[0], x_coord[2]]
        else:
            index = [3*i, 3*i+1]
            coord = [x_coord[0], x_coord[1]]
        coeff1 = (p[index[0], 0]*coord[1] - p[index[1], 0]*coord[0]) \
            / (coord[1] - coord[0])
        coeff3 = (p[index[1], 0] - p[index[0], 0]) / (coord[1] - coord[0])
        coeff2 = p[3*i, 1] - coeff3*y_coord[0]
        # coefficients corresponding to the current element
        interp_p[3*i : 3*i+3] = [coeff1 + coeff3*mesh.elem_center[i, 0],
            coeff2 + coeff3*mesh.elem_center[i, 1], coeff3]

    return interp_p


def flux_convert_hybrid(mesh, p):
    """
    Conversion of flux in terms of the coefficients of local barycentric coordinates.

    Keyword arguments
    -----------------
        - mesh      the domain triangulation
        - p         coefficients corresponding to the global interpolation
                    of the flux using edge basis functions

    Return
    ------
        Coefficients of the local barycentric coordinates of each triangles.
    """

    # number of elements
    num_elems = mesh.num_elems

    # pre-allocation of the coefficient vector
    cnvrt_p = np.zeros((3*num_elems, 2), dtype=np.float)

    for i in range(num_elems):
        # current element
        elem = mesh.tri[i, :] - 1
        # coefficients of p at the current element
        c = p[3*i : 3*i+3]
        # nodes of the the element
        pt1 = mesh.node[elem[0], :]
        pt2 = mesh.node[elem[1], :]
        pt3 = mesh.node[elem[2], :]
        # x components of the coefficient vector at the element
        cnvrt_p[3*i : 3*i+3, 0] \
            = [c[0] + c[2]*(pt1[0] - mesh.elem_center[i, 0]),
            c[0] + c[2]*(pt2[0] - mesh.elem_center[i, 0]),
            c[0] + c[2]*(pt3[0] - mesh.elem_center[i, 0])]
        # y components of the coefficient vector at the element
        cnvrt_p[3*i : 3*i+3, 1] \
            = [c[1] + c[2]*(pt1[1] - mesh.elem_center[i, 1]),
            c[1] + c[2]*(pt2[1] - mesh.elem_center[i, 1]),
            c[1] + c[2]*(pt3[1] - mesh.elem_center[i, 1])]

    return cnvrt_p


def cg(A, b, prm, x=None):
    """
    Solve the linear system Ax = b, where A is a symmetric positive definite matrix
    using conjugate gradient method.

    Keyword arguments
    -----------------
        - A     symmetric positive definite matrix
        - b     vector for the right hand side of the linear system
        - prm   parameters for the conjugate gradient method
        - x     initial point

    Return
    ------
        A dictionary corresponding to the approximate solution of the linear system
        Ax = b and the number of iterations.
    """

    if x is None:
        x = np.zeros((len(b),), dtype=np.float)
        r = b
    else:
        r = b - A * x

    if prm["cgmaxit"] is None:
        prm["cgmaxit"] = 3 * len(x)

    p = r
    rho = np.dot(r, r)
    rtol = (prm["cgtol"] * np.linalg.norm(b)) ** 2
    it = 0
    while rho > rtol and it < prm["cgmaxit"]:
        it = it + 1
        if it > 1:
            beta = rho / rho_old
            p = r + beta * p
        q = A * p
        alpha = rho / np.dot(p, q)
        x = x + alpha * p
        r = r - alpha * q
        rho_old = rho
        rho = np.dot(r,r)

    if it == prm["cgmaxit"] and rho > rtol:
        print("CG WARNING: Maximum number of iterations reached"
              +"  without satisfying tolerance.")

    return dict(sol=x, nit=it)


def pcg(A, b, P_inv, prm, x=None):
    """
    Solve the linear system Ax = b, where A is a symmetric positive definite matrix
    using preconditioned conjugate gradient method.

    Keyword arguments
    -----------------
        - A         symmetric positive definite matrix
        - b         vector for the right hand side of the linear system
        - P_inv     inverse of the preconditioner
        - prm       parameters for the conjugate gradient method
        - x         initial point

    Return
    ------
        A dictionary corresponding to the approximate solution of the linear system
        Ax = b and the number of iterations.
    """

    if x is None:
        x = np.zeros((len(b),), dtype=np.float)
        r = b
    else:
        r = b - A * x

    if prm["cgmaxit"] is None:
        prm["cgmaxit"] = 6 * len(x)

    bnorm = np.linalg.norm(b)
    if bnorm == 0:
        bnorm = 1
    rel_error = np.linalg.norm(r) / bnorm
    it = 0
    while rel_error > prm["cgtol"] and it < prm["cgmaxit"]:
        it = it + 1
        z = P_inv * r
        rho = np.dot(r, z)
        if it > 1:
            beta = rho / rho_old
            p = z + beta * p
        else:
            p = z
        q = A * p
        alpha = rho / np.dot(p, q)
        x = x + alpha * p
        r = r - alpha * q
        rho_old = rho
        rel_error = np.linalg.norm(r) / bnorm

    if it == prm["cgmaxit"] and rel_error > prm["cgtol"]:
        print("CG WARNING: Maximum number of iterations reached"
              +"  without satisfying tolerance.")

    return dict(sol=x, nit=it)


def init_p(x, y):
    """
    Initial pressure in the wave equation.
    """

    PI2 = 2 * PI
    return np.sin(PI2 * x) * np.sin(PI2 * y)


def grad_p(x, y):
    """
    Gradient of the initial pressure.
    """

    PI2 = 2 * PI
    px = PI2 * np.cos(PI2 * x) * np.sin(PI2 * y)
    py = PI2 * np.sin(PI2 * x) * np.cos(PI2 * y)
    return px, py


def delta_p(x, y):
    """
    Negative Laplacian of the initial pressure.
    """

    PI2 = 2 * PI
    return 2 * (PI2 ** 2) * init_p(x,y)


def time_coeff_p(t):
    """
    Time-coefficient for the pressure.
    """

    return np.cos(PI * t)


def time_coeff_v(t):
    """
    Time-coefficient for the velocity.
    """

    return (1 / PI) * (1 + np.sin(PI * t))


def dtime_coeff_p(t):
    """
    Derivative of the time-coefficient for the pressure.
    """

    return - PI * np.sin(PI * t)

def desired_p(t, x, y):
    """
    Desired pressure in the optimal control problem.
    """

    p0 = init_p(x, y).reshape(len(x), 1)
    pt = time_coeff_p(t).reshape(1, len(t))
    return np.dot(p0, pt)


def desired_v(t, mesh):
    """
    Desired velocity in the optimal control problem.
    """

    v0 = flux_interp(mesh)
    v0 = v0.reshape(len(v0), 1)
    vt = time_coeff_v(t).reshape(1, len(t))
    return np.dot(v0, vt)


def desired_v_hybrid(t, mesh, transformation):
    """
    Desired velocity in the optimal control problem.
    """

    v0 = flux_interp(mesh)
    v0 = flux_convert(mesh, transformation, v0)
    v0 = flux_interp_hybrid(mesh, v0)
    v0 = v0.reshape(len(v0), 1)
    vt = time_coeff_v(t).reshape(1, len(t))
    return np.dot(v0, vt)

def exact_p(t, x, y):
    """
    Exact pressure.
    """

    return desired_p(t, x, y)


def exact_v(t, mesh):
    """
    Exact velocity.
    """

    return desired_v(t, mesh)


def exact_v_hybrid(t, mesh, transformation):
    """
    Exact velocity in the hybrid formulation.
    """

    return desired_v_hybrid(t, mesh, transformation)


def exact_source(t, x, y):
    """
    Exact right hand side of the wave equation.
    """

    p0 = init_p(x, y).reshape(len(x), 1)
    f0 = delta_p(x, y).reshape(len(x), 1)
    pt = dtime_coeff_p(t).reshape(1, len(t))
    ft = time_coeff_v(t).reshape(1, len(t))
    return np.dot(p0, pt) + np.dot(f0, ft)


def time_coeff_dual_p(t):
    """
    Time-coefficient for the dual pressure.
    """

    return - np.sin(PI * t)


def time_coeff_dual_v(t):
    """
    Time-coefficient for the dual velocity.
    """

    return (1 / PI) * (np.cos(PI * t) + 1)


def dtime_coeff_dual_p(t):
    """
    Derivative of the time-coefficient for the dual pressure.
    """

    return - PI * np.cos(PI * t)


def exact_dual_p(t, x, y):
    """
    Exact pressure of the adjoint wave equation.
    """

    p0 = init_p(x, y).reshape(len(x), 1)
    pt = time_coeff_dual_p(t).reshape(1, len(t))
    return np.dot(p0, pt)


def exact_dual_v(t, mesh):
    """
    Exact velocity of the adjoint wave equation.
    """

    v0 = flux_interp(mesh)
    v0 = v0.reshape(len(v0), 1)
    vt = time_coeff_dual_v(t).reshape(1, len(t))
    return np.dot(v0, vt)


def exact_dual_v_hybrid(t, mesh, transformation):
    """
    Exact velocity of the adjoint wave equation.
    """

    v0 = flux_interp(mesh)
    v0 = flux_convert(mesh, transformation, v0)
    v0 = flux_interp_hybrid(mesh, v0)
    v0 = v0.reshape(len(v0), 1)
    vt = time_coeff_dual_v(t).reshape(1, len(t))
    return np.dot(v0, vt)


def exact_dual_source(t, x, y):
    """
    Exact right hand side of the adjoint wave equation.
    """

    p0 = init_p(x, y).reshape(len(x), 1)
    f0 = delta_p(x, y).reshape(len(x), 1)
    pt = dtime_coeff_dual_p(t).reshape(1, len(t))
    ft = time_coeff_dual_v(t).reshape(1, len(t))
    return - (np.dot(p0, pt) + np.dot(f0, ft))


def init_lag_mul(mesh):
    """
    Initial Lagrange multiplier in the hybridized RT0-FE, which corresponds to the trace
    of initial pressure. The projection is computed using the trapezoidal rule.
    """

    pt1 = mesh.node[mesh.edge[mesh.int_edge, 0]-1, :]
    pt2 = mesh.node[mesh.edge[mesh.int_edge, 1]-1, :]

    lambda_0 = 0.5 * (init_p(pt1[:, 0], pt1[:, 1])
        + init_p(pt2[:, 0], pt2[:, 1]))
    return lambda_0


class Wave_Vars:
    """
    The variables of the Poisson equation.
    """

    def __init__(self, pre, vel, lag=None):
        """
        Class initialization/construction.

        Attributes
        ----------
            - pre       pressure
            - vel       velocity
            - lag       Lagrange multipliers
        """

        self.pre = pre
        self.vel = vel
        self.lag = lag

    def __sub__(self, other):
        if self.lag is not None and other.lag is not None:
            self.lag = self.lag - other.lag
        else:
            pass
        return Wave_Vars(self.pre - other.pre,
                            self.vel - other.vel, self.lag)

    def __mul__(self, c):
        if self.lag is not None:
            self.lag = c * self.lag
        else:
            pass
        return Wave_Vars(c * self.pre, c * self.vel, self.lag)


def pre_norm_P1(pre, area, area_kron, dt, Mat):
    """
    L2-norm with respect to piecewise linear basis functions.
    """

    J = np.dot(pre[:, 0], Mat.P1 * area_kron * pre[:, 0])
    for i in range(pre.shape[1] - 1):
        J = J + np.dot(pre[:, i] + pre[:, i+1], Mat.P1 * area_kron * pre[:, i+1])

    return np.sqrt(dt * J / 6)


def vel_norm(vel, dt, Mat):
    """
    L2-norm with respect to the lowest order Raviart-Thomas finite elements.
    """

    J = np.dot(vel[:, 0], Mat.mss * vel[:, 0])
    for i in range(vel.shape[1] - 1):
        J = J + np.dot(vel[:, i] + vel[:, i+1], Mat.mss * vel[:, i+1])

    return np.sqrt(dt * J / 6)


def MFEM_State_Solver(init, q, Mat, num_time_step, dt, area, area_inv, prm):
    """
    Solves the state equation.
    """

    num_elems = init.pre.shape[0]
    num_edges = init.vel.shape[0]
    p = np.zeros((num_elems, num_time_step+1), dtype=np.float)
    v = np.zeros((num_edges, num_time_step+1), dtype=np.float)
    p[:, 0] = init.pre
    v[:, 0] = init.vel

    for i in range(num_time_step):
        qsum = q[:, i] + q[:, i+1]
        rhs = Mat.M_minus * v[:, i] - dt * Mat.stf * p[:, i] \
            - 0.25 * (dt**2) * Mat.stf * qsum
        v[:, i+1] = cg(Mat.M_plus, rhs, prm)["sol"]
        p[:, i+1] = p[:, i] + (dt/2) * qsum \
            + (dt/2) * area_inv * Mat.stf.T * (v[:, i] + v[:, i+1])

    if prm["pressure_ave"]:
        p[:, 1:] = 0.5 * (p[:, :-1] + p[:, 1:])
    if prm["velocity_ave"]:
        v[:, 1:] = 0.5 * (v[:, :-1] + v[:, 1:])

    return Wave_Vars(p, v)


def MFEM_Residual(state, desired):
    """
    Computes the difference between the state and the desired state.
    """

    dp = np.kron([1, 1, 1], state.pre).reshape(3*state.pre.shape[0],
        state.pre.shape[1]) - desired.pre
    dv = state.vel - desired.vel

    return Wave_Vars(dp, dv)


def MFEM_Adjoint_Solver(residual, Mat, num_time_step, dt, area, area_inv, prm):
    """
    Solves the adjoint equation.
    """

    num_elems = Mat.stf.shape[1]
    num_edges = residual.vel.shape[0]
    w = np.zeros((num_elems, num_time_step+1), dtype=np.float)
    y = np.zeros((num_edges, num_time_step+1), dtype=np.float)

    for i in range(num_time_step, -1, -1):
        if i == num_time_step:
            rhs = 0.25 * prm["alpha"] * (dt**2) \
                * Mat.stf * Mat.P0P1 * residual.pre[:, i] \
                + prm["beta"] * (dt/2) * Mat.mss * residual.vel[:, i]
            y[:, i] = cg(Mat.M_plus, rhs, prm)["sol"]
            w[:, i] = prm["alpha"] * (dt/2) * Mat.P0P1 * residual.pre[:, i] \
                - (dt/2) * area_inv * Mat.stf.T * y[:, i]
        elif i == 0:
            w[:, i] = w[:, i+1] \
                - (dt/2) * area_inv * Mat.stf.T * y[:, i+1] \
                + (dt/2) * prm["alpha"] * Mat.P0P1 * residual.pre[:, i]
            rhs = (dt/2) * Mat.stf * w[:, i+1]
            y[:, i] = y[:, i+1] + prm["beta"] * (dt/2) * residual.vel[:, i] \
                + (1 / Mat.mss.diagonal()) * rhs
        else:
            rhs = Mat.M_minus * y[:, i + 1] + dt * Mat.stf * w[:, i+1] \
                + 0.5 * (dt**2) * prm["alpha"] * Mat.stf \
                * Mat.P0P1 * residual.pre[:, i] \
                + prm["beta"] * dt * Mat.mss * residual.vel[:, i]
            y[:, i] = cg(Mat.M_plus, rhs, prm)["sol"]
            w[:, i] = w[:, i+1] \
                + prm["alpha"] * dt * Mat.P0P1 * residual.pre[:, i] \
                - (dt/2) * area_inv * Mat.stf.T * (y[:, i] + y[:, i+1])

    if prm["pressure_ave"]:
        w[:, :-1] = 0.5 * (w[:, :-1] + w[:, 1:])
    if prm["velocity_ave"]:
        y[:, :-1] = 0.5 * (y[:, :-1] + y[:, 1:])

    return Wave_Vars(w, y)


def MFEM_Adjoint_to_Control(adjoint):
    """
    Maps the adjoint to the control.
    """

    return adjoint.pre


def MFEM_Cost(residual, q, Mat, num_time_step, dt, area, area_kron, prm):
    """
    Computes the cost.
    """

    J = prm["alpha"] \
        * np.dot(residual.pre[:, 0], area_kron * residual.pre[:, 0]) \
        + prm["beta"] \
        * np.dot(residual.vel[:, 0], Mat.mss * residual.vel[:, 0]) \
        + prm["gamma"] * np.dot(q[:, 0], area * q[:, 0])

    for i in range(num_time_step):
        sum_pre = residual.pre[:, i] + residual.pre[:, i+1]
        sum_vel = residual.vel[:, i] + residual.vel[:, i+1]
        sum_ctr = q[:, i] + q[:, i+1]

        J = J + prm["alpha"] \
            * np.dot(sum_pre, area_kron * residual.pre[:, i+1]) \
            + prm["beta"] * np.dot(sum_vel, Mat.mss * residual.vel[:, i+1]) \
            + prm["gamma"] * np.dot(sum_ctr, area * q[:, i+1])

    return dt * J / 6


def MFEM_Cost_Derivative(control, adj_to_ctrl, prm):
    """
    Computes the derivative of the cost functional.
    """

    return prm["gamma"] * control + adj_to_ctrl


def MFEM_Optimality_Residual(residual, area, num_time_step, dt):
    """
    Calculates the residual norm of the optimality condition in the hybrid formulation.
    """

    R = np.dot(residual[:, 0], area * residual[:, 0])
    for i in range(num_time_step):
        sum_res = residual[:, i] + residual[:, i+1]
        R = R + np.dot(sum_res, area * residual[:, i+1])

    return np.sqrt(dt * R / 6)


def HFEM_PostProcess(state, Mat, mesh):
    """
    Post-process the lagrange multiplier to pressure.
    """

    lag_extend = np.zeros((mesh.num_edges, state.lag.shape[1]), dtype=np.float)
    lag_extend[mesh.int_edge, :] = state.lag

    return Mat.postproc * (Mat.extend * lag_extend)


def HFEM_State_Solver(init, q, Mat, num_time_step, dt, area, area_inv, prm):
    """
    Solves the state equation in the hybrid formulation.
    """

    num_elems = init.pre.shape[0]
    p = np.zeros((num_elems, num_time_step+1), dtype=np.float)
    v = np.zeros((3*num_elems, num_time_step+1), dtype=np.float)
    lag = np.zeros((init.lag.shape[0], num_time_step+1), dtype=np.float)
    p[:, 0] = init.pre
    v[:, 0] = init.vel
    lag[:, 0] = init.lag

    for i in range(num_time_step):
        qsum = q[:, i] + q[:, i+1]
        rhs = Mat.M_minus * v[:, i] - dt * Mat.stf * p[:, i] \
            - 0.25 * (dt**2) * Mat.stf * qsum
        v[:, i+1] = pcg(Mat.M_plus, rhs, Mat.P_inv, prm)["sol"]
        p[:, i+1] = p[:, i] + (dt/2) * qsum \
            + (dt/2) * area_inv * Mat.stf.T * (v[:, i] + v[:, i+1])
        lag[:, i+1] = Mat.lag.T * (v[:, i+1] + v[:, i]) / prm["pen"] \
            - lag[:, i]

    if prm["pressure_ave"]:
        p[:, 1:] = 0.5 * (p[:, :-1] + p[:, 1:])
    if prm["velocity_ave"]:
        v[:, 1:] = 0.5 * (v[:, :-1] + v[:, 1:])
    if prm["lagrange_ave"]:
        lag[:, 1:] = 0.5 * (lag[:, :-1] + lag[:, 1:])

    return Wave_Vars(p, v, lag)


def HFEM_Residual(state, desired, Mat, mesh):
    """
    Computes the difference between the state and the desired state in the hybrid
    formulation.
    """

    pre_postproc = HFEM_PostProcess(state, Mat, mesh)

    return Wave_Vars(pre_postproc - desired.pre, state.vel - desired.vel)


def HFEM_Adjoint_Solver(residual, Mat, num_time_step, dt, area, area_inv, prm):
    """
    Solves the adjoint equation in the hybrid formulation.
    """

    num_elems = Mat.stf.shape[1]
    num_edges = residual.vel.shape[0]
    w = np.zeros((num_elems, num_time_step+1), dtype=np.float)
    y = np.zeros((3*num_elems, num_time_step+1), dtype=np.float)
    mu = np.zeros((Mat.lag.shape[1], num_time_step+1), dtype=np.float)

    for i in range(num_time_step, -1, -1):
        if i == num_time_step:
            rhs = 0.25 * prm["alpha"] \
                * (dt**2) * Mat.stf * Mat.P0P1 * residual.pre[:, i] \
                + prm["beta"] * (dt/2) * Mat.mss * residual.vel[:, i]
            y[:, i] = pcg(Mat.M_plus, rhs, Mat.P_inv, prm)["sol"]
            w[:, i] = prm["alpha"] * (dt/2) * Mat.P0P1 * residual.pre[:, i] \
                - (dt/2) * area_inv * Mat.stf.T * y[:, i]
            mu[:, i] = - Mat.lag.T * y[:, i] / prm["pen"]
        elif i == 0:
            w[:, i] = w[:, i+1] - (dt/2) * area_inv * Mat.stf.T * y[:, i+1] \
                + (dt/2) * prm["alpha"] * Mat.P0P1 * residual.pre[:, i]
            rhs = (dt/2) * Mat.stf * w[:, i+1] \
                + (dt/2) * Mat.lag * mu[:, i+1]
            y[:, i] = y[:, i+1] + prm["beta"] * (dt/2) * residual.vel[:, i] \
                + (1 / Mat.mss.diagonal()) * rhs
            mu[:, i] = - (dt/2) * mu[:, i+1] \
                - (dt/2) * Mat.lag.T * y[:, i+1] / prm["pen"]
        else:
            rhs = Mat.M_minus * y[:, i+1] + dt * Mat.stf * w[:, i+1] \
                + 0.5 * (dt**2) * prm["alpha"] \
                * Mat.stf * Mat.P0P1 * residual.pre[:, i] \
                + prm["beta"] * dt * Mat.mss * residual.vel[:, i]
            y[:, i] = pcg(Mat.M_plus, rhs, Mat.P_inv, prm)["sol"]
            w[:, i] = w[:, i+1] + prm["alpha"] * dt \
                * Mat.P0P1 * residual.pre[:, i] \
                - (dt/2) * area_inv * Mat.stf.T * (y[:, i] + y[:, i+1])
            mu[:, i] = - Mat.lag.T * (y[:, i] + y[:, i+1]) / prm["pen"] \
                - mu[:, i+1]

    if prm["pressure_ave"]:
        w[:, :-1] = 0.5 * (w[:, :-1] + w[:, 1:])
    if prm["velocity_ave"]:
        y[:, :-1] = 0.5 * (y[:, :-1] + y[:, 1:])
    if prm["lagrange_ave"]:
        mu[:, :-1] = 0.5 * (mu[:, :-1] + mu[:, 1:])

    return Wave_Vars(w, y, mu)


def HFEM_Adjoint_to_Control(adjoint, Mat, mesh):
    """
    Maps the adjoint to control in the hybrid formulation.
    """

    return HFEM_PostProcess(adjoint, Mat, mesh)


def HFEM_Cost(residual, q, Mat, num_time_step, dt, area, area_kron, prm):
    """
    Computes the cost in the hybrid formulation.
    """

    J = prm["alpha"] \
        * np.dot(residual.pre[:, 0], Mat.P1 * area_kron * residual.pre[:, 0]) \
        + prm["beta"] * np.dot(residual.vel[:, 0],
            Mat.mss * residual.vel[:, 0]) \
        + prm["gamma"] * np.dot(q[:, 0], Mat.P1 * area_kron * q[:, 0])

    for i in range(num_time_step):

        sum_pre = residual.pre[:, i] + residual.pre[:, i+1]
        sum_vel = residual.vel[:, i] + residual.vel[:, i+1]
        sum_ctr = q[:, i] + q[:, i+1]

        J = J + prm["alpha"] \
            * np.dot(sum_pre, Mat.P1 * area_kron * residual.pre[:, i+1]) \
            + prm["beta"] * np.dot(sum_vel, Mat.mss * residual.vel[:, i+1]) \
            + prm["gamma"] * np.dot(sum_ctr, Mat.P1 * area_kron * q[:, i+1])

    return dt * J / 6


def HFEM_Optimality_Residual(Mat, residual, area, area_kron, num_time_step, dt):
    """
    Calculates the residual norm of the optimality condition in the hybrid formulation.
    """

    R = np.dot(residual[:, 0], Mat.P1 * area_kron * residual[:, 0])
    for i in range(num_time_step):
        sum_res = residual[:, i] + residual[:, i+1]
        R = R + np.dot(sum_res, Mat.P1 * area_kron * residual[:, i+1])

    return np.sqrt(dt * R / 6)


def build_desired(mesh, transformation, prm, t):
    """
    Assembles the desired states for the optimal control problem.
    """

    x = mesh.node[mesh.tri-1, 0].reshape(3*mesh.num_elems,)
    y = mesh.node[mesh.tri-1, 1].reshape(3*mesh.num_elems,)
    pd = desired_p(t, x, y)
    if not prm["hybrid"]:
        vd = desired_v(t, mesh)
    else:
        vd = desired_v_hybrid(t, mesh, transformation)
    return Wave_Vars(pd, vd)


def sparse_matrix_condition_number(A):
    """
    Returns the largest eigenvalue, smallest eigenvalue, and condition number of the
    sparse matrix A.
    """
    eigs_lm = eigs(A, which="LM", return_eigenvectors=False, k=1)[0].real
    eigs_sm = eigs(A, which="SM", return_eigenvectors=False, k=1,
        tol=1e-10)[0].real
    return eigs_lm, eigs_sm, eigs_lm / eigs_sm


class OCP:
    """
    The optimal control problem class.
    """

    def __init__(self, prm=None, desired=None, mesh=None):
        """
        Class initialization.
        """
        if mesh is None:
            self.mesh = square_uni_trimesh(prm["n"])
        else:
            self.mesh = mesh
        self.t = np.linspace(0, prm["T"], prm["num_time_step"]+1)
        self.dt = self.t[1] - self.t[0]
        self.transformations = affine_transform(self.mesh)
        self.area = sp.diags(self.transformations["det"] / 2.0)
        self.area_inv = sp.diags(2.0 / self.transformations["det"])
        self.area_kron = sp.kron(np.identity(3), self.area)
        self.prm = default_parameters()
        if prm is not None:
            self.prm.update(prm)
        self.Mat = Matrices(self.mesh, self.prm, self.dt, self.area,
            self.area_inv, self.transformations)
        self.desired = build_desired(
            self.mesh, self.transformations, self.prm, self.t)
        if not self.prm["hybrid"]:
            self.init_control \
                = np.zeros((self.mesh.num_elems, prm["num_time_step"]+1),
                dtype=np.float)
        else:
            self.init_control \
                = np.zeros((3*self.mesh.num_elems, prm["num_time_step"]+1),
                dtype=np.float)
        self.rhs = None
        self.init = Wave_Vars(self.Mat.P0P1*self.desired.pre[:, 0],
            self.desired.vel[:, 0], init_lag_mul(self.mesh))

    def state_solver(self, control):
        if self.rhs is not None:
            if not self.prm["hybrid"]:
                control = self.Mat.P0P1 * self.rhs + control
            else:
                control = control + self.rhs
        if not self.prm["hybrid"]:
            return MFEM_State_Solver(self.init, control, self.Mat,
                self.prm["num_time_step"], self.dt, self.area, self.area_inv,
                self.prm)
        else:
            control = self.Mat.P0P1 * control
            return HFEM_State_Solver(self.init, control, self.Mat,
                self.prm["num_time_step"], self.dt, self.area, self.area_inv,
                self.prm)

    def residual(self, state):
        if not self.prm["hybrid"]:
            return MFEM_Residual(state, self.desired)
        else:
            return HFEM_Residual(state, self.desired, self.Mat, self.mesh)

    def adjoint_solver(self, residual):
        if not self.prm["hybrid"]:
            return MFEM_Adjoint_Solver(residual, self.Mat,
                self.prm["num_time_step"], self.dt, self.area, self.area_inv,
                self.prm)
        else:
            return HFEM_Adjoint_Solver(residual, self.Mat,
                self.prm["num_time_step"], self.dt, self.area, self.area_inv,
                self.prm)

    def der_cost(self, control, adj_to_ctrl):
        return MFEM_Cost_Derivative(control, adj_to_ctrl, self.prm)

    def cost(self, residual, control):
        if not self.prm["hybrid"]:
            return MFEM_Cost(residual, control, self.Mat,
                self.prm["num_time_step"], self.dt, self.area,
                self.area_kron, self.prm)
        else:
            return HFEM_Cost(residual, control, self.Mat,
                self.prm["num_time_step"], self.dt, self.area,
                self.area_kron, self.prm)

    def adjoint_to_control(self, adjoint):
        if not self.prm["hybrid"]:
            return MFEM_Adjoint_to_Control(adjoint)
        else:
            return HFEM_Adjoint_to_Control(adjoint, self.Mat, self.mesh)

    def denom_init_step(self, state, control):
        if not self.prm["hybrid"]:
            return MFEM_Cost(state, control, self.Mat,
                self.prm["num_time_step"], self.dt, self.area,
                self.area_kron, self.prm)
        else:
            state.pre = np.kron([1, 1, 1], state.pre).reshape(
                3*state.pre.shape[0], state.pre.shape[1])
            return HFEM_Cost(state, control, self.Mat,
                self.prm["num_time_step"], self.dt, self.area,
                self.area_kron, self.prm)

    def optimality_residual(self, residual):
        if not self.prm["hybrid"]:
            return MFEM_Optimality_Residual(residual,
                self.area, self.prm["num_time_step"], self.dt)
        else:
            return HFEM_Optimality_Residual(self.Mat, residual,
                self.area, self.area_kron, self.prm["num_time_step"], self.dt)


def Barzilai_Borwein(ocp, SecondPoint="None", info=True, version=1):
    """
    Barzilai-Borwein version of the gradient method.

    The algorithm stops if the consecutive cost function values have relative error less
    than the pescribed tolerance or the maximum number of iterations is reached.

    Keyword arguments
    -----------------
        - ocp           a class for the optimal control problem
        - info          Prints the iteration number, cost value and relative
                        error of consecutive cost values. (default True).
        - version       Either 1, 2, or 3. Method of getting the steplength.
                        Let dc and dj be the residues of the control and the
                        derivative of the cost functional, and s be the
                        steplength. The following are implemented depending
                        on the value of version:
                        If <version==1> then
                            s = (dc,dj) / (dj,dj).
                        If <version==2> then
                            s = (dc,dc) / (dc,dj).
                        If <version==3> then
                            s = (dc,dj) / (dj,dj) if the iteration number is
                            odd and s = (dc,dc) / (dc,dj) otherwise.
                        Here, (,) represents the inner product in Rn.
                        The default value of version is set to 1.
        - SecondPoint   The second point of the gradient method. If value is
                        <None> then the second point is given by
                        x = x - g/|g| where x is the initial point and g is
                        its gradient value. If value is <"LS"> then the
                        second point is calculated via inexact line search
                        with Armijo steplenght criterion.

    Return
    ------
        The list of state, control, adjoint and residual variables
        of the optimal control problem.

    Notes
    -----
        The ocp class should have at least the following methods:
        <state_solver>
            A function that solves the state equation.
        <adjoint_solver>
            A function that solves the adjoint equation.
        <residual>
            A function that computes the difference between
            the state and the desired state.
        <cost>
            The cost functional.
        <der_cost>
            Derivative of the cost functional.
        <adjoint_to_control>
            A function that maps the adjoint to the control.
        <optimality_residual>
            A function that computes the measure on which the
            necessary condition is satisfied, that is, norm of
            gradient is small.
        <denom_init_step>
            A function that calculates the denominator in the
            steepest descent steplength.
    """

    COST_VALUES = []
    OPT_RESIDUE = []

    if info:
        string = strcolor.cyan_b("\nBARZILAI-BORWEIN GRADIENT METHOD"
                  + "\t\tTolerance = {:.1e}\t\tVersion {}\n")
        print(string.format(ocp.prm["ocptol"], version))

    # main algorithm
    start_time = time.time()
    for i in range(ocp.prm["ocpmaxit"]):
        if i == 0:
            if info:
                print("Iteration: 1")
            state = ocp.state_solver(ocp.init_control)
            residue = ocp.residual(state)
            cost_old = ocp.cost(residue, ocp.init_control)
            adjoint = ocp.adjoint_solver(residue)
            control_old = ocp.init_control
            control = ocp.init_control \
                - ocp.der_cost(ocp.init_control,
                ocp.adjoint_to_control(adjoint))
            if SecondPoint == "LS":
                num = np.sum(control * control)
                steplength = num / (2 * ocp.denom_init_step(state, control))
                control = steplength * control
                state = ocp.state_solver(control)
                residue = ocp.residual(state)
                cost = ocp.cost(residue, control)
                alpha = 1
                iters = 0
                while cost > cost_old - (1e-4) * alpha * num:
                    alpha = alpha * 0.5
                    control = alpha * control
                    state = state * alpha
                    residue = ocp.residual(state)
                    cost = ocp.cost(residue, control)
                    iters = iters + 1
                if info:
                    print("Number of Backtracking Iterations: " + str(iters))
            elif SecondPoint == None:
                state = ocp.state_solver(control)
                residue = ocp.residual(state)
                cost = ocp.cost(residue, control)
                steplength = 1.0
            try:
                cost
            except UnboundLocalError:
                message = ("Undefined option: Either of the following:"
                           + " <None> or 'LS' is implemented.")
                warnings.warn(message, UserWarning)
                break
            if info:
                string = "\tCost Value = {:.10e}"
                print(string.format(cost))
                string = ("\tSteplength = {:.10e}"
                    + "\t\tOptimality Res = {:.10e}")
                res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
                opt_res = ocp.optimality_residual(res)
                print(string.format(steplength, opt_res))
                COST_VALUES += [cost]
                OPT_RESIDUE += [opt_res]
        else:
            if info:
                print("Iteration: {}".format(i+1))
            adjoint_old = ocp.adjoint_to_control(adjoint)
            adjoint = ocp.adjoint_solver(residue)
            control_residue = control - control_old
            adjoint_residue = ocp.adjoint_to_control(adjoint) - adjoint_old
            res_dercost = ocp.der_cost(control_residue, adjoint_residue)
            if version == 1:
                steplength = np.sum(control_residue * res_dercost) \
                    / np.sum(res_dercost * res_dercost)
            elif version == 2:
                steplength = np.sum(control_residue * control_residue) \
                    / np.sum(control_residue * res_dercost)
            elif version == 3:
                if (i % 2) == 1:
                    steplength = np.sum(control_residue * res_dercost) \
                        / np.sum(res_dercost * res_dercost)
                else:
                    steplength = np.sum(control_residue * control_residue) \
                        / np.sum(control_residue * res_dercost)
            control_old = control
            control = control \
                - steplength \
                * ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
            state = ocp.state_solver(control)
            cost_old = cost
            residue = ocp.residual(state)
            cost = ocp.cost(residue, control)
            rel_error = np.abs(cost - cost_old) / cost
            if info:
                string = ("\tCost Value = {:.10e}"
                    + "\t\tRelative Error = {:.10e}")
                print(string.format(cost, rel_error))
                string = ("\tSteplength = {:.10e}"
                    + "\t\tOptimality Res = {:.10e}")
                res = ocp.der_cost(control, ocp.adjoint_to_control(adjoint))
                opt_res = ocp.optimality_residual(res)
                print(string.format(steplength, opt_res))
                COST_VALUES += [cost]
                OPT_RESIDUE += [opt_res]
            if rel_error < ocp.prm["ocptol"]:
                if info:
                    print("Optimal solution found.")
                break
    if i == ocp.prm["ocpmaxit"] and rel_error > ocp.prm["ocptol"]:
        print("BB Warning: Maximum number of iterations reached"
            + " without satisfying the tolerance.")
    end_time = time.time()
    if info:
        print(strcolor.green_b("\nElapsed time is " + "{:.8f}".format(
            end_time-start_time) + " seconds.\n"))

    return {"state": state, "adjoint": adjoint, "control": control,
            "residue": residue, "costvalues": COST_VALUES,
            "optresidue": OPT_RESIDUE}


def print_line():
    """
    Prints a line.
    """

    print("-"*90)


def print_start():
    """
    Prints machine platform and python version.
    """

    print(strcolor.green_b("*"*90 + "\n"))
    start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(strcolor.green_b("Start of Run: " + start + "\n"))

    string = strcolor.green_b(
        "PYTHON VERSION: {} \nPLATFORM: {} \nPROCESSOR: {}"
        + "\nVERSION: {} \nMAC VERSION: {} \nMAX FREQUENCY: {} MHz"
        + "\nMEMORY INFO: "
        + "Total = {} GB, Available = {:.2f} GB, Used = {:.2f} GB, Percentage = {:.2f}%")
    print(string.format(sys.version, platform.platform(),
        platform.uname()[5], platform.version()[:60]
        + "\n" + platform.version()[60:], platform.mac_ver(),
        psutil.cpu_freq().max, psutil.virtual_memory().total / 1024**3,
        psutil.virtual_memory().available / 1024**3,
        psutil.virtual_memory().used / 1024**3,
        psutil.virtual_memory().percent) + "\n")


def print_end():
    """
    Prints end datetime of execution.
    """

    end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(strcolor.green_b("\nEnd of Run: " + end + "\n"))
    print(strcolor.green_b("*"*90 + "\n"))


def print_details(mesh, Mat, prm, dt):
    """
    Prints the details on the mesh, matrices, and parameters.
    """

    print(strcolor.cyan_b("="*90))
    if prm["hybrid"]:
        print(strcolor.cyan_b(
            "\t\t\tHybrid FEM for Optimal Control of Wave Equation"))
    else:
        print(strcolor.cyan_b(
            "\t\t\tMixed FEM for Optimal Control of Wave Equation"))
    print(strcolor.cyan_b("="*90))
    print(strcolor.cyan_b("\nMESH DETAILS"))
    print("\tNumber of Nodes:                   {}".format(mesh.num_nodes))
    print("\tNumber of Elements:                {}".format(mesh.num_elems))
    print("\tNumber of Edges:                   {}".format(mesh.num_edges))
    print("\tSpatial Meshsize:                  {:.10f}".format(mesh.size()))
    print("\tNumber of Time Nodes:              {}".format(prm["num_time_step"]))
    print("\tTemporal Meshsize:                 {:.10f}".format(dt))

    print(strcolor.cyan_b("\nMATRIX DETAILS"))
    print("\tSize of Mass Matrix:               ({}, {})"
        .format(Mat.mss.shape[0], Mat.mss.shape[1]))
    print("\tSize of Stiffness Matrix:          ({}, {})"
        .format(Mat.stf.shape[0], Mat.stf.shape[1]))
    if Mat.lag is not None:
        print("\tSize of Lagrange Matrix:           ({}, {})"
         .format(Mat.lag.shape[0], Mat.lag.shape[1]))
    print("\tSize of System Matrix:             ({}, {})"
        .format(Mat.M_plus.shape[0], Mat.M_plus.shape[1]))
    print("\tSystem Matrix Density:             {:.10e}"
        .format(sparse_matrix_density(Mat.M_plus)))
    if Mat.lag is None:
        Mat.EIGMAX, Mat.EIGMIN, Mat.CONDNO \
            = sparse_matrix_condition_number(Mat.M_plus)
    else:
        Mat.EIGMAX, Mat.EIGMIN, Mat.CONDNO \
            = sparse_matrix_condition_number(sp.diags(Mat.P_inv) * Mat.M_plus)
    print("\tSystem Matrix Condition Number:    {:.10e}".format(Mat.CONDNO))
    print("\t               Max Eigenvalue:     {:.10e}".format(Mat.EIGMAX))
    print("\t               Min Eigenvalue:     {:.10e}".format(Mat.EIGMIN))

    print(strcolor.cyan_b("\nCOEFFICIENTS IN THE COST FUNCTIONAL"))
    print("\talpha (pressure)                   {:.1e}".format(prm["alpha"]))
    print("\tbeta  (velocity)                   {:.1e}".format(prm["beta"]))
    print("\tgamma (control)                    {:.1e}".format(prm["gamma"]))

    print(strcolor.cyan_b("\nPARAMETERS"))
    print("\tCG Tolerance                       {:.1e}".format(prm["cgtol"]))
    print("\tBB Tolerance                       {:.1e}".format(prm["ocptol"]))
    if Mat.lag is not None:
        print("\tPenalty Term                       {:.1e}".format(prm["pen"]))
    print("\tPressure Averaged                  {}".format(prm["pressure_ave"]))
    print("\tVelocity Averaged                  {}".format(prm["velocity_ave"]))
    if Mat.lag is not None:
        print("\tLagrange Averaged                  {}\n".format(prm["lagrange_ave"]))
    else:
        print("")

    print_line()
