# Math libraries
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg

# Progress bar
from tqdm import tqdm, trange


def get_coord_from_vertex(v, grid_shape):
    """Get the coordinates of a grid vertex from its index."""
    v = np.array(v)
    grid_shape = np.array(grid_shape)
    prod = np.cumprod(grid_shape[::-1])[::-1][1:][:, None]
    first_coord = (v // prod) % grid_shape[:-1][:, None]
    last_coord = v % grid_shape[-1]
    return np.vstack([first_coord, last_coord]).T


def get_vertex_from_coord(coord, grid_shape):
    """Get the index of a grid vertex from its coordinates."""
    coord = np.array(coord)
    grid_shape = np.array(grid_shape)
    prod = np.cumprod(grid_shape[::-1])[::-1][1:][:, None]
    return (coord[:, :-1].dot(prod) + coord[:, -1:])[:, 0]


def prec_conj_grad(A, b, init, M_1, channel=0, iterations=25):
    """Perform preconditioned conjugate gradient descent."""
    x = init
    r = b - A.dot(x)
    d = M_1(r)
    delta_new = r.dot(d)
    for it in trange(iterations, desc="Conjugate gradient - channel {}".format(channel)):
        q = A.dot(d)
        alpha = delta_new / d.dot(q)
        x = x + alpha * d
        r = r - alpha * q
        s = M_1(r)
        delta_old = delta_new
        delta_new = r.dot(s)
        beta = delta_new / delta_old
        d = s + beta * d
    return x


def M_jacobi(y, A):
    print("jac")
    return y / A.diagonal()


def bilateral_representation(V, sigma):
    """Compute a bilateral splat matrix for a matrix V of abstract pixels positions."""
    D = V.shape[1]
    grid_shape = np.ceil((V.max(axis=0) / sigma) + 1).astype(int)

    n_abstract_pixels = len(V)
    n_vertices = np.prod(grid_shape)

    pos_grid = np.rint(V / sigma).astype(int)

    # Positions of the nearest neighbor vertices associated with each abstract pixel
    nearest_neighbors = get_vertex_from_coord(pos_grid, grid_shape)
    # Vertices that are nearest neighbor to at least one pixel
    # (all the other vertices of the grid are useless)
    useful_vertices = np.sort(np.unique(nearest_neighbors))
    n_useful_vertices = len(useful_vertices)

    # Dictionary of indices for useful vertices
    useful_vertex_to_ind = np.empty(n_vertices)
    useful_vertex_to_ind[useful_vertices] = np.arange(n_useful_vertices)

    # Construction of the splat matrix: (vertex, abstract pixel) => neighbor?
    row_ind = useful_vertex_to_ind[nearest_neighbors]
    col_ind = np.arange(n_abstract_pixels)
    data = np.ones_like(row_ind)
    S = sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(n_useful_vertices, n_abstract_pixels)
    )

    # Positions of the useful vertices from the grid
    new_V = get_coord_from_vertex(useful_vertices, grid_shape)

    return S, new_V


def build_pyramid(useful_vertex_to_coord):
    """Construct a pyramid of ever coarser splat matrices."""
    tqdm(desc="Pyramid space construction")
    V = useful_vertex_to_coord
    S_pyr = []
    while len(V) > 1:
        Sk, V = bilateral_representation(V, 2 * np.ones(V.shape[1]))
        S_pyr.append(Sk)
    return S_pyr


def build_P(S_pyr):
    """Deduce the hierarchical projection matrix from the pyramid of splat matrices."""
    prod = sparse.eye(S_pyr[0].shape[1])
    P = prod
    for s in S_pyr:
        prod = s.dot(prod)
        P = sparse.vstack([P, prod])
    return P


def build_z_weight(S_pyr, alpha, beta):
    """Compute weights for all the stages of the pyramid space."""
    z_weight = np.ones(S_pyr[0].shape[1])
    for k, s in enumerate(S_pyr):
        z_weight = np.hstack([
            z_weight,
            (alpha ** (- beta - k - 1)) * np.ones(s.shape[0])
        ])
    return z_weight


def M_hier(y, A, P, z_weight):
    """Compute hierarchical preconditioner."""
    z_size, y_size = P.shape

    P1 = P.dot(np.ones(y_size))
    Py = P.dot(y)
    PA = P.dot(A.diagonal())

    return P.T.dot(z_weight * P1 * Py / PA)


def y_hier(S, C, T, P, z_weight):
    """Compute hierarchical initialization."""
    z_size, y_size = P.shape

    P1 = P.dot(np.ones(y_size))
    PSc = P.dot(S.dot(C))
    PSct = P.dot(S.dot(C * T))

    y_init = (
            P.T.dot(z_weight * PSct / P1) /
            P.T.dot(z_weight * PSc / P1)
    )

    return y_init


def solve(bilateral_grid, C, T, lambd, precond_init_method="hierarchical", channel=0):
    """Solve a least squares problem
    from its bistochastized splat-blur-slice decomposition
    using the preconditioned conjugate gradient."""
    # Retrieve information from the bilateral grid object
    S, B = bilateral_grid.S, bilateral_grid.B
    Dn, Dm = bilateral_grid.Dn, bilateral_grid.Dm

    # Compute the coefficients of the least-squares problem min Ax^2 + bx + c
    A = lambd * (Dm - Dn.dot(B).dot(Dn)) + sparse.diags(S.dot(C))
    b = S.dot(C * T)
    c = 0.5 * (C * T).dot(T)

    # Apply chosen preconditioning and initialization
    if precond_init_method == "simple":
        # Define initial vector and preconditioning function
        y_init = S.dot(C * T) / np.clip(S.dot(C), a_min=1, a_max=None)

        def M_1(y):
            return M_jacobi(y, A)

    elif precond_init_method == "hierarchical":
        # Retrieve pyramid information from the bilateral grid object
        P = bilateral_grid.P
        z_weight_init = bilateral_grid.z_weight_init
        z_weight_precond = bilateral_grid.z_weight_precond
        # Define initial vector and preconditioning function
        y_init = y_hier(S, C, T, P, z_weight_init)

        def M_1(y):
            return M_hier(y, A, P, z_weight_precond)

    else:
        raise ValueError("Wrong preconditioning")

    # Compute the optimal solution
    y_opt = prec_conj_grad(A, b, init=y_init, M_1=M_1, channel=channel)
    return y_opt


def box_filter_recursive_sparse(I, ct, sigma_H):
    """Apply a recursive box filter with sparse matrices for faster computation."""
    a = np.exp(-np.sqrt(2) / sigma_H)
    d = np.diff(ct)
    J = np.empty_like(I)

    dim, channels = I.shape

    A_forward = sparse.diags([1] + list(1 - a ** d))
    B_forward = sparse.identity(dim) - sparse.diags(a ** d, -1)

    A_backward = sparse.diags(list(1 - a ** d) + [1])
    B_backward = sparse.identity(dim) - sparse.diags(a ** d, 1)

    for channel in range(channels):
        J[:, channel] = spalg.spsolve(B_forward, A_forward.dot(I[:, channel]))
        J[:, channel] = spalg.spsolve(B_backward, A_backward.dot(J[:, channel]))

    return J


def smooth_cols(I, sigma_s, sigma_r, it, I_ref=None, N_it=3):
    """Apply a bilateral filter on all columns of an image."""
    if I_ref is None:
        I_ref = I
    # Compute the current spatial std of the filter
    sigma_H = sigma_s * np.sqrt(3) * (2 ** (N_it - it)) / np.sqrt(4 ** N_it - 1)
    # Intialize the new image
    new_I = np.empty_like(I)
    # Compute the vertical spatial derivative of the image channels
    I_ref_prime = np.vstack([I_ref[:1, :, :], np.diff(I_ref, axis=0)])
    for col in trange(I.shape[1], desc="Domain transform - iteration {} - columns".format(it)):
        # Compute the domain transform
        ct = np.cumsum(1 + (sigma_s / sigma_r) * np.abs(I_ref_prime[:, col, :].sum(axis=1)))
        # Apply the bilateral box filter
        new_I_slice = box_filter_recursive_sparse(I[:, col, :], ct, sigma_H)
        # Fill the column of the new image
        new_I[:, col, :] = new_I_slice
    return new_I


def smooth_rows(I, sigma_s, sigma_r, it, I_ref=None, N_it=3):
    """Apply a bilateral filter on all rows of an image."""
    if I_ref is None:
        I_ref = I
    sigma_H = sigma_s * np.sqrt(3) * (2 ** (N_it - it)) / np.sqrt(4 ** N_it - 1)
    # Intialize the new image
    new_I = np.empty_like(I)
    # Compute the horizontal spatial derivative of the image channels
    I_ref_prime = np.hstack([I_ref[:, :1, :], np.diff(I_ref, axis=1)])
    for row in trange(I.shape[0], desc="Domain transform - iteration {} - rows".format(it)):
        # Compute the domain transform
        ct = np.cumsum(1 + (sigma_s / sigma_r) * np.abs(I_ref_prime[row, :, :].sum(axis=1)))
        # Apply the bilateral box filter
        new_I_slice = box_filter_recursive_sparse(I[row, :, :], ct, sigma_H)
        # Fill the row of the new image
        new_I[row, :, :] = new_I_slice
    return new_I


def domain_transform(I0, sigma_s, sigma_r, I_ref=None, N_it=3):
    """Apply the domain transform to I0 with spatial std sigma_s and value std sigma_r."""
    I = I0.copy()
    for it in range(1, N_it + 1):
        for axis in [0, 1]:
            if axis == 1:
                I = smooth_rows(I, sigma_s, sigma_r, it, I_ref=I_ref, N_it=N_it)
            else:
                I = smooth_cols(I, sigma_s, sigma_r, it, I_ref=I_ref, N_it=N_it)
    return I
