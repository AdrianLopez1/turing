# Contains all high-level functions needed for 
# forward problems of growth, RD, and plotting

import pyvista as pv
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys


def grow_forward():
    
    '''
    Grows mesh by recalculating points at each point in time.
    Calculates Laplace-Beltrami operator at each point in time. (costly)
    '''

    from mesh_laplacian import compute_mesh_laplacian

    grow = False # whether to calculate a new laplacian at each point in time
                # or just copy them all from initial point


    grow_from_rule = False # True = grow via assigned growth rule
                          # False = read growth from file

    if grow:

        if grow_from_rule:
            # growth params
            growth_rate = 1.001

            # initialize points 
            pts = np.zeros((nx, 3, niter+1))
            pts[:,:,0] = mesh.points.copy()

        else:
            # reads points from file
            pts = np.load('pts.npy')

        # initialize laplacians
        laps = np.zeros((nx, nx, niter+1))
        laps[:,:,0] = compute_mesh_laplacian(mesh)


        # run growth loop
        print("Beginning growth loop...")
        for i in range(niter):
            sys.stdout.write("\rIteration {0}/{1} ({2}%)".format(i+1, niter, int(100*(i+1)/niter)))
            sys.stdout.flush()

            if grow_from_rule:
                # update points according to growth rule
                # pts[:,:,i+1] = pts[:,:,i] * np.random.uniform(1.0, growth_rate, (nx, 3)) # randomly growing isotropically

                # pts[:,:,i+1] = pts[:,:,i] 
                # pts[:,2,i+1] = pts[:,2,i] *  np.random.uniform(1.0, growth_rate, nx) # randomly growing in z direction 

                pts = pts[:,:,0] * growth_rate * dt * i # uniform additive isotropic growth

            mesh.points = pts[:,:,i+1]

            # calculate laplacian
            laps[:,:,i+1] = compute_mesh_laplacian(mesh)

        # reset mesh points
        mesh.points = pts[:,:,0]

    else:
        # initialize points 
        pts = np.zeros((nx, 3, niter+1))
        pts[:,:,0] = mesh.points.copy()

        # copy initial points to all times
        pts = np.stack((pts[:,:,0],) * (niter+1), axis=2)


        # initialize laplacians
        laps = np.zeros((nx, nx, niter+1))
        laps[:,:,0] = compute_mesh_laplacian(mesh)

        # copy initial laplacian to all times
        laps = np.stack((laps[:,:,0],) * (niter+1), axis=2)

    print("\nGrowth loop completed.")

    # write results to disk
    np.save('laps.npy', laps)
