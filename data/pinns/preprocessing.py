'''
@author: Yongji Wang
Goal: "preprocessing.py" normalize the synthetic data from COMSOL and
organize the data into a form that is required for the PINN training
'''

import numpy as np
import jax.numpy as jnp


def normalize_data(data):
    '''
    :param data: original dataset
    :return X_smp, U_smp, X_ct, n_ct, data_info
    '''

    # extract the velocity data
    xraw = data['xd']   # unit [m] position
    yraw = data['yd']   # unit [m] position
    uraw = data['ud']   # unit [m/s] ice velocity
    vraw = data['vd']   # unit [m/s] ice velocity

    # extract the thickness data (may have different position)
    xraw_h = data['xd_h']   # unit [m] position
    yraw_h = data['yd_h']   # unit [m] position
    hraw = data['hd']       # unit [m] ice thickness

    # extract the position of the calving front (right side of the domain)
    xct = data['xct']    # unit [m] position
    yct = data['yct']    # unit [m] position
    nnct = data['nnct']  # unit vector

    #%%

    # flatten the velocity data into 1d array
    x0 = xraw.flatten()
    y0 = yraw.flatten()
    u0 = uraw.flatten()
    v0 = vraw.flatten()

    # flatten the thickness data into 1d array
    x0_h = xraw_h.flatten()
    y0_h = yraw_h.flatten()
    h0 = hraw.flatten()

    # remove the nan value in the velocity data
    idxval_u = jnp.where(~np.isnan(u0))[0]
    x = x0[idxval_u, None]
    y = y0[idxval_u, None]
    u = u0[idxval_u, None]
    v = v0[idxval_u, None]

    # remove the nan value in the thickness data
    idxval_h = jnp.where(~np.isnan(h0))[0]
    x_h = x0_h[idxval_h, None]
    y_h = y0_h[idxval_h, None]
    h = h0[idxval_h, None]

    #%%
    # calculate the mean and range of the domain
    x_mean = jnp.mean(x)
    x_range = (x.max() - x.min()) / 2
    y_mean = jnp.mean(y)
    y_range = (y.max() - y.min()) / 2

    # calculate the mean and std of the velocity
    u_mean = jnp.mean(u)
    u_range = jnp.std(u) * 2
    v_mean = jnp.mean(v)
    v_range = jnp.std(v) * 2

    # calculate the mean and std of the thickness
    h_mean = jnp.mean(h)
    h_range = jnp.std(h) * 2

    # normalize the velocity data
    x_n = (x - x_mean) / x_range
    y_n = (y - y_mean) / y_range
    u_n = (u - u_mean) / u_range
    v_n = (v - v_mean) / v_range

    # normalize the thickness data
    xh_n = (x_h - x_mean) / x_range
    yh_n = (y_h - y_mean) / y_range
    h_n = (h) / h_mean

    # normalize the calving front position
    xct_n = (xct - x_mean) / x_range
    yct_n = (yct - y_mean) / y_range

    # group the raw data
    data_raw = [x0, y0, u0, v0, x0_h, y0_h, h0]
    # group the normalized data
    data_norm = [x_n, y_n, u_n, v_n, xh_n, yh_n, h_n]
    # group the nan info of original data
    idxval_all = [idxval_u, idxval_h]
    # group the shape info of original data
    dsize_all = [uraw.shape, hraw.shape]

    # group the mean and range info for each variable (shape = (5,))
    data_mean = jnp.hstack([x_mean, y_mean, u_mean, v_mean, h_mean])
    data_range = jnp.hstack([x_range, y_range, u_range, v_range, h_range])

    # gathering all the data information
    data_info = [data_mean, data_range, data_norm, data_raw, idxval_all, dsize_all]

    #%% data grouping

    # group the input and output into matrix
    X_star = [jnp.hstack((x_n, y_n)), jnp.hstack((xh_n, yh_n))]
    X_ct = jnp.hstack((xct_n, yct_n))
    # sequence of output matrix column is u,v,h
    U_star = [jnp.hstack((u_n, v_n)), h_n]

    return X_star, U_star, X_ct, nnct, data_info

def normalize_data_simple(x_data,z_data,w_data,x_bc_div,z_bc_div,x_bc_bed,z_bc_bed,x_bc_surf,z_bc_surf,u_surf):
    # make sure the bc information is passed as a vector, not a mesh
    # extract the velocity data
    xraw = x_data   # unit [m] position
    zraw = z_data   # unit [m] position
    wraw = w_data  # unit [m/s] ice velocity

    # flatten the velocity data into 1d array
    x0 = xraw.flatten()
    z0 = zraw.flatten()
    w0 = wraw.flatten()

    x0_surf = x_bc_surf.flatten()
    z0_surf = z_bc_surf.flatten()
    u0 = u_surf.flatten()

    x0_bed = x_bc_bed.flatten()
    z0_bed = z_bc_bed.flatten()

    # remove the nan value in the velocity data
    idxval_w = jnp.where(~np.isnan(w0))[0]
    x = x0[idxval_w, None]
    z = z0[idxval_w, None]
    w = w0[idxval_w, None]

    idxval_u = jnp.where(~np.isnan(u0))[0]
    x_surf = x0_surf[idxval_u, None]
    z_surf = z0_surf[idxval_u, None]
    u = u0[idxval_u, None]

    # calculate the mean and range of the domain
    x_mean = jnp.mean(x)
    x_range = (x.max() - x.min()) / 2
    z_mean = jnp.mean(z)
    z_range = (z.max() - z.min()) / 2

    # calculate the mean and std of the velocity
    w_mean = jnp.mean(w)
    w_range = jnp.std(w) * 2

    # normalize the velocity data
    x_n = (x - x_mean) / x_range
    z_n = (z - z_mean) / z_range
    w_n = (w - w_mean) / w_range

    # normalize the boundary data coords
    x_div_n = (x_bc_div - x_mean) / x_range
    z_div_n = (z_bc_div - z_mean) / z_range
    x_surf_n = (x_surf - x_mean) / x_range
    z_surf_n = (z_surf - z_mean) / z_range
    x_bed_n = (x0_bed.reshape((len(x0_bed),1)) - x_mean) / x_range
    z_bed_n = (z0_bed.reshape((len(z0_bed),1)) - z_mean) / z_range
    
    # normalize the surface velocities using w_range
    u_surf_n = u / w_range

    # group the raw data
    data_raw = [x0, z0, w0]
    # group the normalized data
    data_norm = [x_n, z_n, w_n]
    # group the nan info of original data
    idxval_all = idxval_w
    # group the shape info of original data
    dsize_all = wraw.shape

    # group the mean and range info for each variable (shape = (3,))
    data_mean = jnp.hstack([x_mean, z_mean, w_mean])
    data_range = jnp.hstack([x_range, z_range, w_range])

    # gathering all the data information
    data_info = [data_mean, data_range, data_norm, data_raw, idxval_all, dsize_all]

    # group the input and output into matrix
    X_star = jnp.hstack((x_n, z_n))
    X_bc = [jnp.hstack((x_div_n,z_div_n)),jnp.hstack((x_bed_n,z_bed_n)),jnp.hstack((x_surf_n,z_surf_n))]
    # sequence of output matrix column is 
    U_star = [w_n]

    return X_star, U_star, X_bc, u_surf_n, data_info

def normalize_data_masscon_real(x_grid,z_grid,zeta_i_grid,w_i_grid,x_bed,z_bed,x_surf,z_surf,u_surf):
    # make sure the bc information is passed as a vector, not a mesh
    # extract the velocity data
    xraw = x_grid   # unit [m] position
    zraw = z_grid
    zeta_raw = zeta_i_grid   # unit [m] position
    wraw = w_i_grid  # unit [m/s] ice velocity

    # flatten the velocity data into 1d array
    x0 = xraw.flatten()
    z0 = zraw.flatten()
    w0 = wraw.flatten()

    x0_surf = x_bc_surf.flatten()
    z0_surf = z_bc_surf.flatten()
    u0 = u_surf.flatten()

    x0_bed = x_bc_bed.flatten()
    z0_bed = z_bc_bed.flatten()
