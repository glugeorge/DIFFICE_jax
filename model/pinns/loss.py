import jax.numpy as jnp


# define the mean squared error
def ms_error(diff):
    return jnp.mean(jnp.square(diff), axis=0)


#%% loss for inferring isotropic viscosity

def loss_iso_create(predf, eqn_all, scale, lw):
    ''' a function factory to create the loss function based on given info
    :param predf: neural network function for solutions
    :param eqn_all: governing equation and boundary conditions
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the velocity data and their position
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]

        # load the thickness data and their position
        xh_smp = data['smp'][2]
        h_smp = data['smp'][3]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:2]
        h_pred = net(xh_smp)[:, 2:3]

        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err))
        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        bd_err = ms_error(f_bd)

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)

        # load the loss_ref
        loss_ref = loss_fun.lref
        # calculate the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd) / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                data_err, eqn_err, bd_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    return loss_fun


#%% loss for inferring anisotropic viscosity

def loss_aniso_create(predf, eqn_all, scale, lw):
    ''' a function factory to create the loss function based on given info
    :param predf: neural network function for solutions
    :param eqn_all: governing equation and boundary conditions
    :return: a loss function (callable)
    '''

    # separate the governing equation and boundary conditions
    gov_eqn, front_eqn = eqn_all

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the data of normalization condition
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]
        xh_smp = data['smp'][2]
        h_smp = data['smp'][3]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_bd = data['bd'][0]
        nn_bd = data['bd'][1]

        # calculate the gradient of phi at origin
        output = net(x_smp)
        u_pred = output[:, 0:2]
        mu_pred = output[:, 3:4]
        eta_pred = output[:, 4:5]
        h_pred = net(xh_smp)[:, 2:3]

        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_bd, term_bd = front_eqn(net, x_bd, nn_bd, scale)

        # calculate the mean squared root error of normalization cond.
        data_u_err = ms_error(u_pred - u_smp)
        data_h_err = ms_error(h_pred - h_smp)
        data_err = jnp.hstack((data_u_err, data_h_err))
        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        bd_err = ms_error(f_bd)
        # calculate the difference between mu and eta
        sp_err = ms_error((jnp.sqrt(mu_pred) - jnp.sqrt(eta_pred)) / 2)

        # set the weight for each condition and equation
        data_weight = jnp.array([1., 1., 0.6])
        eqn_weight = jnp.array([1., 1.])
        bd_weight = jnp.array([1., 1.])

        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err * data_weight)
        loss_eqn = jnp.sum(eqn_err * eqn_weight)
        loss_bd = jnp.sum(bd_err * bd_weight)
        loss_sp = jnp.sum(sp_err)

        # load the loss_ref
        loss_ref = loss_fun.lref
        # load the weight for the regularization loss
        wsp = loss_fun.wsp
        # define the total loss
        loss = (loss_data + lw[0] * loss_eqn + lw[1] * loss_bd + wsp * loss_sp) / loss_ref
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd, loss_sp]),
                                data_err, eqn_err, bd_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    loss_fun.wsp = lw[2]
    return loss_fun

def loss_masscon_create(predf, eqn_all, scale, lw):

    # separate the governing equation and boundary conditions
    gov_eqn, bc_div_eqn, bc_bed_eqn, bc_surf_eqn = eqn_all

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the velocity data and their position
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]
        x_smp_ns = data['smp'][2]
        u_smp_ns = data['smp'][3]

        # load the position and weight of collocation points
        x_col = data['col'][0]
        x_col_ns = data['col'][1]

        x_div = data['bc_div']
        x_bed = data['bc_bed']
        x_surf = data['bc_surf']
        u_surf = data['u_surf']

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 1:2] # not 0:2 because we only have data in w
        u_pred_ns = net(x_smp_ns)[:, 1:2]
        
        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_pred_ns, term_ns = gov_eqn(net, x_col_ns, scale)

        f_div = bc_div_eqn(net, x_div)
        f_bed = bc_bed_eqn(net, x_bed, scale)
        f_surf = bc_surf_eqn(net, x_surf)

        # calculate the mean squared root error of normalization cond.
        data_err = ms_error(u_pred - u_smp)
        data_err_ns = ms_error(u_pred_ns - u_smp_ns)

        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        eqn_err_ns = ms_error(f_pred_ns)

        div_err = ms_error(f_div)
        bed_err = ms_error(f_bed)
        surf_err = ms_error(f_surf-u_surf)

        # set weights for boundary conditions
        bd_weight = [1,0.1,10]

        # all errors should be 1d arrays
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err)+jnp.sum(data_err_ns)
        loss_eqn = jnp.sum(eqn_err) + jnp.sum(eqn_err_ns)
        loss_bd = jnp.sum(div_err*bd_weight[0]) + jnp.sum(bed_err*bd_weight[1]) + jnp.sum(surf_err*bd_weight[2])

        loss_ref = loss_fun.lref
        # calculate total loss

        # lw should have 3 weights
        loss = (loss_data + lw[0]*loss_eqn + lw[1]*loss_bd ) / loss_ref
        
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                data_err, eqn_err, div_err, bed_err, surf_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    return loss_fun

def loss_masscon_realdata_create(predf, eqn_all, scale, lw):
    # separate the governing equation and boundary conditions
    gov_eqn, bc_div_eqn, bc_bed_eqn, bc_surf_eqn = eqn_all
    
    # define function for n    
    def n_from_rho(rho):
        return 1 + (1.78-1)*rho/918 # n_i = 1.78
    
    # define function for apparent ice velocity and zeta to zeta_i
    def wzeta_i_from_wzeta(zeta,w,n):
        c = 3e8
        X = 7
        ni = 1.78
        # need to unflatten then flatten n again
        tau0 = (2/c)*0 # some integral function
        D1 = 0 
        D2 = 0
        T = tau0 - (2/c)*D1**2/D2 + (2/c)*jnp.sqrt(D1**4/D2**2 + (X**2/4)*D1/D2)
        zeta_i = T*c/(2*ni)
        w_i = (n/1.78)*w
        return zeta_i, w_i
    
    def data_error(surface,x_smp,obs,pred):
        w_pred = pred[:, 1:2]
        rho_pred = pred[:, 2:3]

        x0,z0 = jnp.split(x_smp,2,axis=1)
        x_surf,z_surf = jnp.split(surface,2,axis=1)

        z_surf_interp = jnp.interp(x0,x_surf,z_surf) # rethink this 
        zeta = z_surf_interp-z0

        x_obs,zeta_i_obs = jnp.split(obs[0],2,axis=1) # x should be scaled, z should not be
        w_i_obs = obs[1]
        n = n_from_rho(rho_pred)

        zeta_i, w_i = wzeta_i_from_wzeta(zeta,w_pred,n)

        w_i_obs_interp = jnp.interp() # some kind of interpolation here

        return ms_error(w_i - w_i_obs_interp)
        
    

    # loss function used for the PINN training
    def loss_fun(params, data): # !!! include w_i, zeta_i data in the dictionary
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the velocity data and their position
        x_smp = data['smp'] # in the sample, only include grid, no u

        obs = data['obs']

        # load the position and weight of collocation points
        x_col = data['col']

        x_div = data['bc_div']
        x_bed = data['bc_bed']
        x_surf = data['bc_surf']
        u_surf = data['u_surf']

        # calculate the gradient of phi at origin
        pred = net(x_smp) # not 0:2 because we only have data in w

        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        f_div = bc_div_eqn(net, x_div)
        f_bed = bc_bed_eqn(net, x_bed, scale)
        f_surf = bc_surf_eqn(net, x_surf)

        # calculate the mean squared root error of normalization cond.
        data_err = data_error(x_surf,x_smp,obs,pred)

        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        div_err = ms_error(f_div)
        bed_err = ms_error(f_bed)
        surf_err = ms_error(f_surf-u_surf)

        # set weights for boundary conditions
        bd_weight = [1,0.1,10]

        # all errors should be 1d arrays
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err )
        loss_eqn = jnp.sum(eqn_err)
        loss_bd = jnp.sum(div_err*bd_weight[0]) + jnp.sum(bed_err*bd_weight[1]) + jnp.sum(surf_err*bd_weight[2])

        loss_ref = loss_fun.lref
        # calculate total loss

        # lw should have 3 weights
        loss = (loss_data + lw[0]*loss_eqn + lw[1]*loss_bd ) / loss_ref
        
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn, loss_bd]),
                                data_err, eqn_err, div_err, bed_err, surf_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    return loss_fun

def loss_momentum_create_synthetic(predf, eqn_all, scale, lw):

    # separate the governing equation and boundary conditions
    gov_eqn, eqn_bc = eqn_all

    # loss function used for the PINN training
    def loss_fun(params, data):
        # create the function for gradient calculation involves input Z only
        net = lambda z: predf(params, z)
        # load the velocity data and their position
        x_smp = data['smp'][0]
        u_smp = data['smp'][1]

        # load the position and weight of collocation points
        x_col = data['col']

        #x_div = data['bc_div']
        #x_bed = data['bc_bed']
        x_bc = data['bc_flanks']
        mu_bc = data['mu_flanks']

        # calculate the gradient of phi at origin
        u_pred = net(x_smp)[:, 0:4] #  0:4 because we have data in u,w, rho, p
        
        # calculate the residue of equation
        f_pred, term = gov_eqn(net, x_col, scale)
        #f_bed = eqn_bc(net,x_bed)
        #f_div = eqn_bc(net,x_div)
        mu_pred = eqn_bc(net,x_bc)

        # calculate the mean squared root error of normalization cond.
        data_err = ms_error(u_pred - u_smp)

        # calculate the mean squared root error of equation
        eqn_err = ms_error(f_pred)
        bc_err = ms_error(mu_pred - mu_bc)
        #div_err = ms_error(f_div)
        #bed_err = ms_error(f_bed)

        # set weights for boundary conditions
        bd_weight = [1,1]

        # all errors should be 1d arrays
        # calculate the overall data loss and equation loss
        loss_data = jnp.sum(data_err)
        loss_eqn = jnp.sum(eqn_err) 
        loss_bd = jnp.sum(bc_err)#jnp.sum(div_err*bd_weight[0]) + jnp.sum(bed_err*bd_weight[1]) 

        loss_ref = loss_fun.lref
        # calculate total loss

        # lw should have 3 weights
        loss = (loss_data + lw[0]*loss_eqn + lw[1]*loss_bd) / loss_ref
        
        # group the loss of all conditions and equations
        loss_info = jnp.hstack([jnp.array([loss, loss_data, loss_eqn,loss_bd]),
                                data_err, eqn_err,bc_err])
        return loss, loss_info

    loss_fun.lref = 1.0
    return loss_fun