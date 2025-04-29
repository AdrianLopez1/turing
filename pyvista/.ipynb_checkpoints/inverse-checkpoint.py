def eval_j(u, v, ut, vt, c1, c2, g1, g2, d1, d2):
    """
    Evaluates the objective functional J.
    """
    
    j = np.sum(g1*(u-ut)**2 + g2*(v-vt)**2)*dx + d1*c1**2 + d2*c2**2 
    j*= 0.5
    
    return j


def descend_c(u_t, v_t, c1, c2, l, jtol):
    '''Find optimal c1, c2 by descending gradient of J'''
    
    # solve SE for one step, starting at target
    u1, v1 = step_se(u_t, v_t, c1, c2)

    # evaluate cost after this timestep
    j1 = eval_j(u1, v1, u_t, v_t, c1, c2, g1, g2, d1, d2)


    for i in range(niter):

        # step c1, c2 and evaluate cost gradient
        increment = 1e-6
        c1s, c2s = c1+increment, c2+increment

        u1s1, v1s1 = step_se(u_t, v_t, c1s, c2)
        u1s2, v1s2 = step_se(u_t, v_t, c1, c2s)

        j1s1 = eval_j(u1s1, v1s1, u_t, v_t, c1s, c2, g1, g2, d1, d2)
        j1s2 = eval_j(u1s2, v1s2, u_t, v_t, c1, c2s, g1, g2, d1, d2)

        jgrad = ((j1s1-j1)/increment, (j1s2-j1)/increment)

        # step both in direction opposite to gradient
        c1_new = c1-l*jgrad[0]
        c2_new = c2-l*jgrad[1]

        # test whether j actually decreased;
        # if not then change step size and do again
        # if yes then update c1, c2

        u1_new, v1_new = step_se(u_t, v_t, c1_new, c2_new)
        j1_new = eval_j(u1_new, v1_new, u_t, v_t, c1_new, c2_new, g1, g2, d1, d2)
        
        print(i, j1_new)

        if j1_new > j1:
            l /= 10

        else:
            l *= 3/2
            c1, c2 = c1_new, c2_new
            u1, u2 = u1_new, v1_new
            j1 = j1_new


        if j1_new < jtol:
            # plt.plot(u1_new)
            print('convergence to optimal solution reached in {} iterations'.format(i))
            break

        if i==niter-1: 
            # plt.plot(u1_new)
            print(j1)
            print('failed to converge after {} iterations'.format(niter))

    return c1, c2
