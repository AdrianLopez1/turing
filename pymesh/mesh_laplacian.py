import numpy as np

def compute_mesh_laplacian(mesh):

    # construct adjacency matrix using points
    adj = np.zeros((mesh.n_points, mesh.n_points), dtype=bool)

    for i in range(mesh.n_points):
        for j in mesh.point_neighbors(i):
            adj[i,j] += 1

    # need angles between edges
    # to find the angles for the edge connecting points i and j, find points adjacent to both i and j
    # call these points a and b (there should always be 2 for a triangular mesh)
    # can find angle cotangent by ratio of cross product to dot product

    cota = np.zeros((mesh.n_points, mesh.n_points))
    cotb = np.zeros((mesh.n_points, mesh.n_points))

    for i in range(mesh.n_points):
        for j in mesh.point_neighbors(i):

            ab = np.where(adj[i,:]*adj[j,:] == 1)[0] # this fails when the surface has a boundary

            aj = mesh.points[ab[0]] - mesh.points[j]
            aj /= np.linalg.norm(aj)
            ai = mesh.points[ab[0]] - mesh.points[i]
            ai /= np.linalg.norm(ai)

            bj = mesh.points[ab[1]] - mesh.points[j]
            bj /= np.linalg.norm(bj)
            bi = mesh.points[ab[1]] - mesh.points[i]
            bi /= np.linalg.norm(bi)

            cota[i,j] = np.dot(aj, ai)/ np.linalg.norm(np.cross(aj,ai))
            cotb[i,j] = np.dot(bj, bi)/ np.linalg.norm(np.cross(bj,bi))


    # Find vertex area
    cell_sizes = mesh.compute_cell_sizes()

    vertex_area = np.zeros(mesh.n_points)
    for i in range(mesh.n_points):
        # for each vertex, find adjacent cells
        adj_cells = mesh.point_cell_ids(i)

        # sum 1/3 of these cells' areas
        # ASSUMES TRIANGULAR MESH!!!
        vertex_area[i] = (1/3)*np.sum(cell_sizes['Area'][adj_cells])


    # Put it all together to get discrete laplacian
    minv = np.diag(1/vertex_area)

    n = np.diag((1/2)*(cota + cotb).sum(axis=1))

    lap_unweighted = (1/2)*(cota + cotb) - n

    lap = minv @ lap_unweighted

    for i in range(mesh.n_points):
        lap[i,:] = lap_unweighted[i,:] / vertex_area[i]
        
    return lap