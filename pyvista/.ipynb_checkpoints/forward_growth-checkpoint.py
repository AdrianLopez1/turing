# Contains all high-level functions needed for 
# forward problems of growth, RD, and plotting

import pyvista as pv
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import sys


def grow_forward(mesh, nx, niter, dt, growth_rate=1.0, grow=False, grow_from_rule=False, input_file=None, output_file=None):
    
    '''
    Grows mesh by recalculating points at each point in time.
    Calculates Laplace-Beltrami operator at each point in time. (costly)
    '''

    from mesh_laplacian import compute_mesh_laplacian

    if grow:

        if grow_from_rule:
            # initialize points 
            pts = np.zeros((nx, 3, niter+1))
            pts[:,:,0] = mesh.points.copy()

        else:
            # reads points from file
            pts = np.load(input_file)

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

                pts[:,:,i+1] = pts[:,:,0]*(1 + growth_rate * dt * (i+1)) # uniform additive isotropic growth

            mesh.points = pts[:,:,i+1]

            # calculate laplacian
            laps[:,:,i+1] = compute_mesh_laplacian(mesh)

        # reset mesh points
        mesh.points = pts[:,:,0]

    else:
        # initialize points 
        pts = np.zeros((nx, 3, niter+1), dtype=np.float32)
        pts[:,:,0] = mesh.points.copy()

        # copy initial points to all times
        pts = np.stack((pts[:,:,0],) * (niter+1), axis=2)


        # initialize laplacians
        laps = np.zeros((nx, nx, niter+1), dtype=np.float32)
        laps[:,:,0] = compute_mesh_laplacian(mesh)

        # copy initial laplacian to all times
        laps = np.stack((laps[:,:,0],) * (niter+1), axis=2)

    print("\nGrowth loop completed.")

    # write results to disk
    np.savez_compressed(output_file, pts=pts, laps=laps)
    
    print("Growth data written to {}.".format(output_file))

    
    
    
def rd_forward(mesh, du, dv, g, a, b, nx, dx, niter, dt, laps, grow=False, output_file=None):
    '''
        Integrates RD DE's forward in time on growing manifold defined by laps.
    '''
    from forward import step_se

    # initialize fields near steady-state solution
    u = np.ones(nx, dtype=float)*(a+b)
    u += np.random.normal(scale=0.01, size=nx)
    v = np.ones(nx, dtype=float)*(b/(a+b)**2)

    u_stored = np.zeros((nx, niter+1))
    u_stored[:,0] = u

    v_stored = np.zeros((nx, niter+1))
    v_stored[:,0] = v


    integrate = True

    if integrate:
        print("Beginning RD integration loop...")
        for i in range(niter):
            sys.stdout.write("\rIteration {0}/{1} ({2}%)".format(i+1, niter, int(100*(i+1)/niter)))
            sys.stdout.flush()

            # Run GMRES to solve for next timestep
            # reference calculated laplacians from growth loop
            if grow: 
                # first apply dilution
                area_current = mesh.area
                mesh.points  = pts[:,:,i+1]
                area_new     = mesh.area
                dilution_factor = area_current / area_new
                
                u *= dilution_factor
                
                # then RD
                u, v = step_se(u,v, a,b,g,du,dv, laps[:,:,i], dx,nx,dt)
            else: u, v = step_se(u,v, a,b,g,du,dv, laps, dx,nx,dt)

            # store for later animation
            u_stored[:,i+1] = u
            v_stored[:,i+1] = v


        print("\nRD loop completed.")
        
    # write results to disk
    # note that this overwrites pts and laps
    # np.savez_compressed(output_file, u=u_stored, v=v_stored)
    
    print("RD data written to {}.".format(output_file))
    
    # reset mesh
    if grow: mesh.points = pts[:,:,0]
    
    
    return u_stored, v_stored




def plot_forward(params, mesh, u_stored, pts, niter, grow=False, mode='dynamic', nskip=1, output_gif='output.gif', cpos='xy', rot=0):
    # run plotting 
    # pts = np.load('pts.npy')

    # Set up plotting
    if mode == 'static':
        p = pv.Plotter(shape=(1,3), notebook=0, off_screen=True)
        # du, dv, g, a, b, nx, dx, niter, dt,
        p.add_text('du={}\ndv={}\ng={}\na={}\nb={}\nnx={}\ndx={}\nniter={}\ndt={}'.format(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8]))


        def plot_mesh(u, subplot):
            mesh.point_data['u'] = u
            p.subplot(0,subplot)
            p.add_mesh(mesh.copy().rotate_y(rot), scalars='u', cmap='gray', show_edges=False)
            p.link_views()
            p.view_isometric()
            p.camera_position = 'xy'

        plot_mesh(u_stored[:,0], 0)

    elif mode == 'dynamic':
        if grow: mesh.points = pts[:,:,-1]
        plotter = pv.Plotter(notebook=False, off_screen=True)
        plotter.add_mesh(
            mesh,
            scalars=u_stored[:,0],
            lighting=False,
            show_edges=True,
            scalar_bar_args={"title": "u"},
            clim=[u_stored.min(), u_stored.max()],
            cmap='hot'
        )
        plotter.camera_position = cpos

        # Open a gif
        plotter.open_gif(output_gif)
        # plotter.camera.zoom(0.8) # have to zoom out to accomodate growth
                                 # would be nice if I could get around this by mode final mesh first



    print("Beginning plotting loop...")
    for i in range(niter):
        sys.stdout.write("\rIteration {0}/{1} ({2}%)".format(i+1, niter, int(100*(i+1)/niter)))
        sys.stdout.flush()
        if mode == 'static':
            if i==int(niter/2):
                # mesh.points = pts[:,:,i]
                plot_mesh(u_stored[:,i], 1)

            if i==int(niter-1):
                # mesh.points = pts[:,:,i]
                plot_mesh(u_stored[:,i], 2)


        elif mode == 'dynamic':
            if i%nskip==0:
                if grow: plotter.update_coordinates(pts[:,:,i], render=False)
                plotter.update_scalars(u_stored[:,i], render=False)

                # Write a frame. This triggers a render.
                plotter.write_frame()

    print("\nPlotting loop completed.")



    # p = pv.Plotter(shape=(1,1), notebook=0)
    # mesh.point_data['turing'] = u
    # p.add_mesh(mesh, scalars='turing', cmap='gray')      
    if mode == 'static': 
        p.remove_scalar_bar()
        p.image_scale = 4
        p.screenshot(output_gif[:-4]+'.png', window_size=(500,400))
        p.show()

        # p.save_graphic(output_gif[:-4]+'.svg')
        # p.save_graphic('test.svg')

    elif mode == 'dynamic': plotter.close()

    print("Plotting completed.")
    # plt.show()
    # print(u, info)