import numpy as np
import scipy.sparse as sparse

# Progress bar
from tqdm import tqdm

from libs.fast_bilateral_solver.utils \
    import get_vertex_from_coord, get_coord_from_vertex, build_pyramid, build_P, build_z_weight


class BilateralGrid:

    def __init__(self, ref_img, sigma):
        """Initialize a bilateral grid with a reference image and a standard deviation vector."""
        self.ref_img = ref_img
        self.sigma = sigma

        self.compute_everything()

    def compute_everything(self):
        """Perform all computations necessary for the fast bilateral solver."""
        self.compute_useful_stuff()
        self.compute_splat()
        self.compute_blur()
        self.compute_bistochastization()
        self.compute_pyramid_space()

    def compute_useful_stuff(self):
        """Translate the pixels of a 2d image into D-dimensional positions."""
        # Spatial coordinates of all the image pixels
        self.x_ref_img = np.indices(self.ref_img.shape[:2])[0].flatten()
        self.y_ref_img = np.indices(self.ref_img.shape[:2])[1].flatten()

        # Positions (coordinates + values) of all the image pixels
        self.pos_ref_img = np.hstack([
            self.x_ref_img[:, None],
            self.y_ref_img[:, None],
            self.ref_img[self.x_ref_img, self.y_ref_img]
        ])

        # Dimension of the position: 2 + number of channels
        self.D = 2 + self.ref_img.shape[2]

        # Shape of the D-dimensional bilateral grid:
        # - sizes of the coordinate axes
        # - sizes of the value axes
        self.grid_shape = np.hstack([
            np.ceil(self.ref_img.shape[:2] / self.sigma[:2]) + 1,
            np.ceil(self.ref_img.max() / self.sigma[2:]) + 1
        ]).astype(int)

        # Number of pixels and vertices
        self.n_pixels = np.prod(self.ref_img.shape[:2])
        self.n_vertices = np.prod(self.grid_shape)

    def compute_splat(self):
        """Compute the splat matrix: links pixels to the associated vertices in the grid."""
        tqdm(desc="Splat matrix computation")

        # Positions of the nearest neighbor vertices associated with each image pixel
        self.pos_grid = np.rint(self.pos_ref_img / self.sigma).astype(int)
        # Indices of the nearest neighbor vertices
        self.nearest_neighbors = get_vertex_from_coord(self.pos_grid, self.grid_shape)

        # Vertices that are nearest neighbor to at least one pixel
        # (all the other vertices of the grid are useless)
        self.useful_vertices = np.sort(np.unique(self.nearest_neighbors))
        self.n_useful_vertices = len(self.useful_vertices)

        # Dictionary of coordinates for useful vertices
        self.useful_vertex_to_coord = get_coord_from_vertex(self.useful_vertices, self.grid_shape)
        # Dictionary of indices for useful vertices
        self.useful_vertex_to_ind = np.empty(self.n_vertices)
        self.useful_vertex_to_ind[self.useful_vertices] = np.arange(self.n_useful_vertices)

        # Record if a given vertex is useful (comes in handy for slicing)
        self.vertex_is_useful = np.zeros(self.n_vertices)
        self.vertex_is_useful[self.useful_vertices] = 1

        # Construction of the splat matrix: (vertex, pixel) => neighbor?
        row_ind = self.useful_vertex_to_ind[self.nearest_neighbors]
        col_ind = np.arange(self.n_pixels)
        data = np.ones_like(row_ind)
        self.S = sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(self.n_useful_vertices, self.n_pixels)
        )

    def compute_blur(self):
        """Compute the blur matrix: superposition of (1, 2, 1) filters on the bilateral grid."""
        B = sparse.lil_matrix((self.n_useful_vertices, self.n_useful_vertices))
        # Fill the diagonal of the blur matrix
        B[np.arange(self.n_useful_vertices), np.arange(self.n_useful_vertices)] = 2 * self.D

        # List all the +-1 coordinate changes possible in D dimensions
        possible_neighbor_steps = [
                                      np.array([0] * dim + [1] + [0] * (self.D - dim - 1))
                                      for dim in range(self.D)
                                  ] + [
                                      np.array([0] * dim + [-1] + [0] * (self.D - dim - 1))
                                      for dim in range(self.D)
                                  ]

        for neighbor_step in tqdm(possible_neighbor_steps, desc="Blur matrix computation"):
            # Compute the +-1 neighbors only for the useful vertices
            neighbors_coord = self.useful_vertex_to_coord + neighbor_step

            # Check whether these neighbors are still in the grid
            neighbors_exist = True
            for dim, dim_size in enumerate(self.grid_shape):
                neighbors_exist = (
                        neighbors_exist &
                        (neighbors_coord[:, dim] >= 0) &
                        (neighbors_coord[:, dim] < dim_size)
                )

            # Select only the vertices whose neighbors are still in the grid
            vertices_with_existing_neighbors = self.useful_vertices[neighbors_exist]
            existing_neighbors_coord = neighbors_coord[neighbors_exist]
            existing_neighbors_vertices = get_vertex_from_coord(existing_neighbors_coord, self.grid_shape)

            # Select only the vertices whose neighbors are also useful
            neighbors_among_useful = self.vertex_is_useful[existing_neighbors_vertices].astype(bool)
            vertices_with_useful_neighbors = vertices_with_existing_neighbors[neighbors_among_useful]
            useful_neighbors_vertices = existing_neighbors_vertices[neighbors_among_useful]

            # Construct splat matrix: (vertex, vertex) => filter coefficient
            row_ind = self.useful_vertex_to_ind[vertices_with_useful_neighbors]
            col_ind = self.useful_vertex_to_ind[useful_neighbors_vertices]
            B[row_ind, col_ind] = 1

        self.B = B.tocsr()

    def compute_bistochastization(self, iterations=20):
        """Compute diagonal bistochastization matrices."""
        tqdm(desc="Bistochastization")
        m = self.S.dot(np.ones(self.S.shape[1]))
        n = np.ones(self.B.shape[1])
        for it in range(iterations):
            new_n = np.sqrt((n * m) / (self.B.dot(n)))
            if np.linalg.norm(new_n - n) < 1e-5:
                break
            else:
                n = new_n
        Dn = sparse.diags(n)
        Dm = sparse.diags(m)

        self.Dn, self.Dm = Dn, Dm

    def compute_pyramid_space(self):
        """Compute pyramidal decomposition."""
        self.S_pyr = build_pyramid(self.useful_vertex_to_coord)
        self.P = build_P(self.S_pyr)
        self.z_weight_init = build_z_weight(self.S_pyr, alpha=4, beta=0)
        self.z_weight_precond = build_z_weight(self.S_pyr, alpha=2, beta=5)
