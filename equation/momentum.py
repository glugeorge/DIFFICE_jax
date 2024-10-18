"""
@author: George Lu
"""

import jax.numpy as jnp
from jax import vjp, vmap, lax

# generate matrix required for vjp for vector gradient
def vgmat(x, n_out, idx=None):
    '''
    :param n_out: number of output variables
    :param idx: indice (list) of the output variable to take the gradient
    '''
    if idx is None:
        idx = range(n_out)
    # obtain the number of index
    n_idx = len(idx)
    # obtain the number of input points
    n_pt = x.shape[0]
    # determine the shape of the gradient matrix
    mat_shape = [n_idx, n_pt, n_out]
    # create the zero matrix based on the shape
    mat = jnp.zeros(mat_shape)
    # choose the associated element in the matrix to 1
    for l, ii in zip(range(n_idx), idx):
        mat = mat.at[l, :, ii].set(1.)
    return mat


# vector gradient of the output with input
def vectgrad(func, x):
    # obtain the output and the gradient function
    sol, vjp_fn = vjp(func, x)
    # determine the mat grad
    mat = vgmat(x, sol.shape[1])
    # calculate the gradient of each output with respect to each input
    grad0 = vmap(vjp_fn, in_axes=0)(mat)[0]
    # calculate the total partial derivative of output with input
    n_pd = x.shape[1] * sol.shape[1]
    # reshape the derivative of output with input
    grad = grad0.transpose(1, 0, 2)
    grad_all = grad.reshape(x.shape[0], n_pd)
    return grad_all, sol

def gov_eqn(net, x, scale):
    dmean, drange = scale[0:2]
    x0, z0, w0, mu0 = drange[0:4]
    def grad1stOrder(net, x):
        grad, sol = vectgrad(net, x)
        # order should u w rho p mu
        u = sol[:, 0:1]
        w = sol[:, 1:2]
        rho = sol[:, 2:3]
        p = sol[:, 3:4]
        mu = sol[:,4:5]

        u_x = grad[:, 0:1]
        u_z = grad[:, 1:2]
        w_x = grad[:, 2:3]
        w_z = grad[:, 3:4]
        # rho_x, rho_z for 4:5, 5:6
        p_x = grad[:, 6:7]
        p_z = grad[:, 7:8]

        term1_1 = mu*u_x
        term12_21 = mu*(u_z/z0 + w_x/x0 )
        term1_3 = p_x

        term2_2 = mu*w_z
        term2_3 = p_z + rho

        return jnp.hstack([term1_1,term12_21,term1_3,term2_2,term2_3])

    func_g = lambda x: grad1stOrder(net, x)
    grad_term, term = vectgrad(func_g, x)
    
    e1term1 = 2*grad_term[:, 0:1]/x0**2 # (term1_1,x)
    e1term2 = grad_term[:,3:4]/z0 # (term12_21,z)
    e1term3 = 910*9.81*z0*term[:,2:3]/(mu0*w0*x0)
    e2term1 = grad_term[:,2:3]/x0 # (term12_21,x)
    e2term2 = 2*grad_term[:,7:8]/z0**2
    e2term3 = 910*9.81*term[:,4:5]/(mu0*w0)

    e1 = e1term1 + e1term2 - e1term3
    e2 = e2term1 + e2term2 - e2term3
    f_eqn = jnp.hstack([e1, e2])
    val_term = jnp.hstack([e1term1, e1term2, e1term3, e2term1, e2term2, e2term3])
    return f_eqn, val_term

def eqn_bc(net,x):
    sol, vjp_fn = vjp(net, x)
    u = sol[:,0:1]
    e1 = u
    return e1 