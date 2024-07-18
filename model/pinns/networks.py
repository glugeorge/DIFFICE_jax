import jax.numpy as jnp

# define the basic formation of neural network
def neural_net(params, x, scl, act_s=0):
    '''
    :param params: weights and biases
    :param x: input data [matrix with shape [N, m]]; m is number of inputs)
    :param sgn:  1 for even function and -1 for odd function
    :return: neural network output [matrix with shape [N, n]]; n is number of outputs)
    '''
    # choose the activation function
    actv = [jnp.tanh, jnp.sin][act_s]
    # normalize the input
    H = x  # input has been normalized
    # separate the first, hidden and last layers
    first, *hidden, last = params
    # calculate the first layers output with right scale
    H = actv(jnp.dot(H, first[0]) * scl + first[1])
    # calculate the middle layers output
    for layer in hidden:
        H = jnp.tanh(jnp.dot(H, layer[0]) + layer[1])
    # no activation function for last layer
    var = jnp.dot(H, last[0]) + last[1]
    return var


# wrapper to create solution function with given domain size
def solu_create(scl=1, act_s=0):
    '''
    :param scale: normalization info
    :return: function of the solution (a callable)
    '''
    def f(params, x):
        # generate the NN
        uvh = neural_net(params[0], x, scl, act_s)
        mu = neural_net(params[1], x, scl, act_s)
        sol = jnp.hstack([uvh, jnp.exp(mu)])
        return sol
    return f

def solu_create_simple(scale,scl=1, act_s=0):
    '''
    :param scale: normalization info
    :return: function of the solution (a callable)
    '''
    dmean, drange = scale[0:2]
    x_m = dmean[0]
    x_scale = drange[0]
    z_m = dmean[1]
    z_scale = drange[1]

    def thickness(x):
        h = -(x*x_scale + x_m)**2/2000 + 1000   
        h_nondim = (h-z_m)/z_scale # divide by zscale
        return h_nondim
    
    def f(params, x):
        x0, z0 = jnp.split(x, 2, axis=1)
        H = thickness(x0)
        pm_rho = params[1]
        # generate the NN
        uw = neural_net(params[0], x, scl, act_s)
        rhoL = 1 + (jnp.exp(pm_rho[0]) - 1) * jnp.exp((-H+z0)/jnp.exp(pm_rho[1]))
        sol = jnp.hstack([uw, rhoL])
        return sol
    return f
