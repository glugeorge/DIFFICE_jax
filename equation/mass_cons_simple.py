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
    lx0, lz0, w0 = drange[0:3]
    wm = dmean[2:3]
    grad, sol = vectgrad(net,x)

    u_x = grad[:,0:1]
    w_z = grad[:,3:4]
    rho_z = grad[:,5:6]
    w = sol[:,1:2]
    rho = sol[:,2:3]

    eterm1 = u_x * lz0/lx0
    eterm2 = (w+wm/w0)*rho_z + rho*w_z
    e1 = eterm1 + eterm2

    val_term = jnp.hstack([eterm1,eterm2])
    return e1, val_term
    
# need 2 different BC eqns
def bc_div_eqn(net,x):
    sol, vjp_fn = vjp(net, x)
    u = sol[:,0:1]
    e1 = u
    return e1

def bc_bed_eqn(net,x,us):
    sol, vjp_fn = vjp(net, x)
    u = sol[:,0:1]
    e1 = u
    # Enforcing BC on rho
    #rho = sol[:,2:3]
    #e1 = rho - 1 
    return e1