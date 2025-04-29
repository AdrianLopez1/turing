import scipy.sparse as sp

def step_se(u, v, a, b, g, du, dv, lap, dx=1, nx=0, dt=0, nsteps=1):
    """
    Steps morphogen concentrations u and v forwards in time
    according to the system equations (SE). (Schnakenberg kinetics)
    """
    
    for i in range(nsteps):
        # nonlinear reaction terms
        # for u: u(n)*u(n+1)*v(n)
        u_nonlin = sp.eye(nx)
        u_nonlin.setdiag(u*v)

        # for v: u(n)^2*v(n+1)
        v_nonlin = sp.eye(nx)
        v_nonlin.setdiag(u**2)
        
        # identity matrix
        iden = sp.eye(nx)


        # linear operators for gmres
        A_u = iden - dt*( du*lap/dx**2 - g*iden + g*u_nonlin )
        b_u = u + dt*g*a

        A_v = iden - dt*( dv*lap/dx**2 - g*v_nonlin )
        b_v = v + dt*g*b


        # Run GMRES to solve for next timestep
        u, info_u = sp.linalg.gmres(A_u, b_u, maxiter=1000)
        v, info_v = sp.linalg.gmres(A_v, b_v, maxiter=1000)

        
        # Should catch a convergence failure here using info_u and info_v
        if info_u>0 or info_v>0:
            print("Warning: GMRES convergence failed")
            
    return u, v



import scipy.sparse as sp

def step_se_gm(u, v, r, mu, a, du, dv, lap, dx=1, nx=0, dt=0, nsteps=1):
    """
    Steps morphogen concentrations u and v forwards in time
    according to the system equations (SE). (Gierer and Meinhardt kinetics)
    
    CURRENTLY NOT WORKING, NEED ALTERNATE NUMERICAL SCHEME (SEE GARVIE)
    """
    
    for i in range(nsteps):
        # nonlinear reaction terms
        # for u: u(n)*u(n+1)*v(n)
        u_nonlin = sp.eye(nx)
        u_nonlin.setdiag(r*u/v - mu)

        # for v: u(n)^2*v(n+1)
        v_nonlin = sp.eye(nx)
        v_nonlin.setdiag(-a)
        
        # identity matrix
        iden = sp.eye(nx)


        # linear operators for gmres
        A_u = iden - dt*( du*lap/dx**2 - iden + u_nonlin )
        b_u = u + dt*r

        A_v = iden - dt*( dv*lap/dx**2 - v_nonlin )
        b_v = v + dt*r*u**2


        # Run GMRES to solve for next timestep
        u, info_u = sp.linalg.gmres(A_u, b_u, maxiter=1000)
        v, info_v = sp.linalg.gmres(A_v, b_v, maxiter=1000)

        
        # Should catch a convergence failure here using info_u and info_v
        if info_u>0 or info_v>0:
            print("Warning: GMRES convergence failed")
            
    return u, v