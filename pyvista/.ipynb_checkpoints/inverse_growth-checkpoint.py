# for inverse problem of growth
# into desired mesh from sphere

import pyvista as pv
import numpy as np

def grow_inverse(mesh, nx, niter):
    '''
        solve simple inverse problem for growth
        to end with mesh starting from sphere
    '''

    # initialize points
    pts = np.zeros((nx, 3, niter+1))
    pts[:,:,0] = mesh.points.copy()


    # get each point's distance from origin
    dist = np.sqrt(np.sum(pts[:,:,0]**2, axis=1)) 

    # construct final points: a sphere, by shrinking by distance factors
    pts[:,:,-1] = np.einsum('ui,u->ui', pts[:,:,0], 1/dist)

    # make final points partway between sphere and full shape to avoid singularities
    pts[:,:,-1] = 0.5*pts[:,:,0] + 0.5*pts[:,:,-1]

    # interpolate points in between
    for i in range(1, niter):
        pts[:,:,i] = (1-i/niter) * pts[:,:,0] + i/niter * pts[:,:,-1]
        # print(i/(niter-1))
        # print(i)

    # reverse time ordering to go from shrinking to growing
    pts_flipped = np.flip(pts, axis=2)

    # write result to disk
    np.save('pts.npy', pts_flipped)
