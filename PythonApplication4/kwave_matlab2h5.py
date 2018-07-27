
import numpy as np
import scipy.io as spio
import math
import h5py
from scipy.interpolate import RegularGridInterpolator

# Load variables from Matlab text file

k = h5py.File(r'C:\Users\Anne\Documents\AWS\toSend\simSettings_20180727_0\kwave.h5')
mat = spio.loadmat(r'C:\Users\Anne\Documents\AWS\toSend\simSettings_20180727_0\datafile.mat', squeeze_me=True)

ux_source_flag = mat['ux_source_flag']
uy_source_flag = mat['uy_source_flag']
uz_source_flag = mat['uz_source_flag']
p_source_flag = mat['p_source_flag']
p0_source_flag = mat['p0_source_flag']
transducer_source_flag = mat['transducer_source_flag']
nonuniform_grid_flag = mat['nonuniform_grid_flag']
nonlinear_flag = mat['nonlinear_flag']
absorbing_flag = mat['absorbing_flag']

Nx = mat['Nx']
Ny = mat['Ny']
Nz = mat['Nz']
dx = mat['dx']
dy = mat['dy']
dz = mat['dz']

c_ref = mat['c_ref']

# Used for calculation only
kx_vec = mat['kx_vec']
ky_vec = mat['ky_vec']
kz_vec = mat['kz_vec']

# Used for calculation only
x_size = mat['x_size']
y_size = mat['y_size']
z_size = mat['z_size']


kx = np.tile(kx_vec,[Nz,Ny,1])
ky = np.swapaxes(np.tile(ky_vec,[Nz,Nx,1]),1,2)
kz = np.swapaxes(np.tile(kz_vec,[Nx,Ny,1]),0,2)

x = kx * x_size * dx / (2*math.pi)
y = ky * y_size * dy / (2*math.pi)
z = kz * z_size * dz / (2*math.pi)

ddx_k_shift_pos_r = mat['ddx_k_shift_pos_r']
ddx_k_shift_neg_r = mat['ddx_k_shift_neg_r']
ddy_k_shift_pos = mat['ddy_k_shift_pos']
ddy_k_shift_neg = mat['ddy_k_shift_neg']
ddz_k_shift_pos = mat['ddz_k_shift_pos']
ddz_k_shift_neg = mat['ddz_k_shift_neg']

ddy_k_shift_pos = np.swapaxes([ddy_k_shift_pos],0,1)
ddy_k_shift_neg = np.swapaxes([ddy_k_shift_neg],0,1)
ddz_k_shift_pos = np.swapaxes([[ddz_k_shift_pos]],0,2)
ddz_k_shift_neg = np.swapaxes([[ddz_k_shift_neg]],0,2)

x_shift_neg_r = mat['x_shift_neg_r']
y_shift_neg_r = mat['y_shift_neg_r']
z_shift_neg_r = mat['z_shift_neg_r']


dt = mat['dt']
Nt = mat['Nt']


pml_x_size = mat['pml_x_size']
pml_y_size = mat['pml_y_size']
pml_z_size = mat['pml_z_size']

pml_x_alpha = mat['pml_x_alpha']
pml_y_alpha = mat['pml_y_alpha']
pml_z_alpha = mat['pml_z_alpha']

pml_x = mat['pml_x']
pml_y = mat['pml_y']
pml_z = mat['pml_z']

pml_x_sgx = mat['pml_x_sgx']
pml_y_sgy = mat['pml_y_sgy']
pml_z_sgz = mat['pml_z_sgz']

c_fat = mat['c_fat']
c_mus = mat['c_mus']
c_endo = mat['c_endo']
rho_fat = mat['rho_fat']
rho_mus = mat['rho_mus']
rho_endo = mat['rho_endo']


maps = spio.loadmat(r'C:\Users\Anne\Documents\AWS\toSend\simSettings_20180727_0\mapfile.mat', squeeze_me=True)

mus = maps['mus']
mus = np.swapaxes(mus,0,1)
fat = maps['fat']
fat = np.swapaxes(fat,0,1)
endo = np.ones((Ny,Nx)) - mus - fat

c0 = c_mus*mus + c_fat*fat + c_endo*endo
rho0 = rho_mus*mus + rho_fat*fat + rho_endo*endo

c0 = np.tile(c0,(Nz,1,1))
rho0 = np.tile(rho0,(Nz,1,1))

set_x = kx_vec * x_size * dx / (2*math.pi)
set_y = ky_vec * y_size * dx / (2*math.pi)
set_z = kz_vec * z_size * dx / (2*math.pi)

interp_x = kx_vec[0:Nx-1] * x_size * dx / (2*math.pi) + dx/2
interp_y = ky_vec[0:Ny-1] * y_size * dy / (2*math.pi) + dy/2
interp_z = kz_vec[0:Nz-1] * z_size * dz / (2*math.pi) + dz/2

my_interpolating_function = RegularGridInterpolator((set_z, set_y, set_x), rho0)

# points to be interpolated
pts_sgx = np.vstack(np.meshgrid(set_z,set_y,interp_x)).reshape(3,-1).T
pts_sgy = np.vstack(np.meshgrid(set_z,interp_y,set_x)).reshape(3,-1).T
pts_sgz = np.vstack(np.meshgrid(interp_z,set_y,set_x)).reshape(3,-1).T

rho0_sgx_sub = my_interpolating_function(pts_sgx)
rho0_sgx = np.ones((Nz,Ny,Nx))
rho0_sgx[0:Nz,0:Ny,0:Nx] = rho0
rho0_sgx[:,:,0:Nx-1] = np.reshape(rho0_sgx_sub,(Ny,Nz,Nx-1)).swapaxes(0,1)

rho0_sgy_sub = my_interpolating_function(pts_sgy)
rho0_sgy = np.ones((Nz,Ny,Nx))
rho0_sgy[0:Nz,0:Ny,0:Nx] = rho0
rho0_sgy[:,0:Ny-1,:] = np.reshape(rho0_sgy_sub,(Ny-1,Nz,Nx)).swapaxes(0,1)


rho0_sgz_sub = my_interpolating_function(pts_sgz)
rho0_sgz = np.ones((Nz,Ny,Nx))
rho0_sgz[0:Nz,0:Ny,0:Nx] = rho0
rho0_sgz[0:Nz-1,:,:] = np.reshape(rho0_sgz_sub,(Ny,Nz-1,Nx)).swapaxes(0,1)


sensor_mask_index = mat['sensor_mask_index']
sensor_mask_type = mat['sensor_mask_type']
Nsens = mat['Nsens']


p_source_index = mat['p_source_index']
p_source_input = mat['p_source_input'].reshape(-1,1)
p_source_mode = mat['p_source_mode']
p_source_many = mat['p_source_many']


hf = h5py.File(r'C:\Users\Anne\Documents\AWS\toSend\simSettings_20180727_0\python.h5', 'w')

# Variables are either u8 (uint64) or f4 (float32)

save_h5(hf,'ux_source_flag',ux_source_flag,'int')
save_h5(hf,'uy_source_flag',uy_source_flag,'int')
save_h5(hf,'uz_source_flag',uz_source_flag,'int')
save_h5(hf,'p_source_flag',p_source_flag,'int')
save_h5(hf,'p0_source_flag',p0_source_flag,'int')
save_h5(hf,'transducer_source_flag',transducer_source_flag,'int')
save_h5(hf,'nonuniform_grid_flag',nonuniform_grid_flag,'int')
save_h5(hf,'nonlinear_flag',nonlinear_flag,'int')
save_h5(hf,'absorbing_flag',absorbing_flag,'int')

save_h5(hf,'Nx',Nx,'int')
save_h5(hf,'Ny',Ny,'int')
save_h5(hf,'Nz',Nz,'int')
save_h5(hf,'Nt',Nt,'int')
save_h5(hf,'dx',dx,'float')
save_h5(hf,'dy',dy,'float')
save_h5(hf,'dz',dz,'float')
save_h5(hf,'dt',dt,'float')

save_h5(hf,'rho0',rho0,'float')
save_h5(hf,'rho0_sgx',rho0_sgx,'float')
save_h5(hf,'rho0_sgy',rho0_sgy,'float')
save_h5(hf,'rho0_sgz',rho0_sgz,'float')

save_h5(hf,'c0',c0,'float')
save_h5(hf,'c_ref',c_ref,'float')

save_h5(hf,'sensor_mask_index',sensor_mask_index,'int')
save_h5(hf,'sensor_mask_type',sensor_mask_type,'int')

save_h5(hf,'p_source_mode',p_source_mode,'int')
save_h5(hf,'p_source_many',p_source_many,'int')
save_h5(hf,'p_source_input',p_source_input,'float')
save_h5(hf,'p_source_index',p_source_index,'int')



# Simulation flags
#hf.create_dataset('ux_source_flag', data=[[[np.uint64(ux_source_flag)]]])
#hf.create_dataset('uy_source_flag', data=[[[np.uint64(uy_source_flag)]]])
#hf.create_dataset('uz_source_flag', data=[[[np.uint64(uz_source_flag)]]])
#hf.create_dataset('p_source_flag', data=[[[np.uint64(p_source_flag)]]])
#hf.create_dataset('p0_source_flag', data=[[[np.uint64(p0_source_flag)]]])
#hf.create_dataset('transducer_source_flag', data=[[[np.uint64(transducer_source_flag)]]])
#hf.create_dataset('nonuniform_grid_flag', data=[[[np.uint64(nonuniform_grid_flag)]]])
#hf.create_dataset('nonlinear_flag', data=[[[np.uint64(nonlinear_flag)]]])
#hf.create_dataset('absorbing_flag', data=[[[np.uint64(absorbing_flag)]]])

#Grid properties
#hf.create_dataset('Nx', data=[[[np.uint64(Nx)]]])
#hf.create_dataset('Ny', data=[[[np.uint64(Ny)]]])
#hf.create_dataset('Nz', data=[[[np.uint64(Nz)]]])
#hf.create_dataset('Nt', data=[[[np.uint64(Nt)]]])
#hf.create_dataset('dx', data=[[[np.float32(dx)]]])
#hf.create_dataset('dy', data=[[[np.float32(dy)]]])
#hf.create_dataset('dz', data=[[[np.float32(dz)]]])
#hf.create_dataset('dt', data=[[[np.float32(dt)]]])

# Medium properties
#hf.create_dataset('rho0', data=np.float32(rho0))
#hf.create_dataset('rho0_sgx', data=np.float32(rho0_sgx))
#hf.create_dataset('rho0_sgy', data=np.float32(rho0_sgy))
#hf.create_dataset('rho0_sgz', data=np.float32(rho0_sgz))


#hf.create_dataset('c0', data=np.float32(c0))
#hf.create_dataset('c_ref', data=[[[np.float32(c_ref)]]])


# Sensor properties
#hf.create_dataset('sensor_mask_index', data=np.uint64(sensor_mask_index.reshape(1,1,-1)))
#hf.create_dataset('sensor_mask_type', data=[[[np.uint64(sensor_mask_type)]]])

# Pressure source terms
#hf.create_dataset('p_source_mode', data=[[[np.uint64(p_source_mode)]]])
#hf.create_dataset('p_source_many', data=[[[np.uint64(p_source_many)]]])
#hf.create_dataset('p_source_index', data=[[np.uint64(p_source_index)]])
#hf.create_dataset('p_source_input', data=[np.float32(p_source_input)])

# k-space and shift variables
hf.create_dataset('ddx_k_shift_pos_r', data=[[np.float32(ddx_k_shift_pos_r)]])
hf.create_dataset('ddx_k_shift_neg_r', data=[[np.float32(ddx_k_shift_neg_r)]])
hf.create_dataset('ddy_k_shift_pos', data=[np.float32(ddy_k_shift_pos)])
hf.create_dataset('ddy_k_shift_neg', data=[np.float32(ddy_k_shift_neg)])
hf.create_dataset('ddz_k_shift_pos', data=np.float32(ddz_k_shift_pos))
hf.create_dataset('ddz_k_shift_neg', data=np.float32(ddz_k_shift_neg))

hf.create_dataset('x_shift_neg_r', data=np.float32(x_shift_neg_r.reshape(1,1,-1)))
hf.create_dataset('y_shift_neg_r', data=np.float32(y_shift_neg_r.reshape(1,-1,1)))
hf.create_dataset('z_shift_neg_r', data=np.float32(z_shift_neg_r.reshape(-1,1,1)))

# PML variables
hf.create_dataset('pml_x_size', data=[[[np.uint64(pml_x_size)]]])
hf.create_dataset('pml_y_size', data=[[[np.uint64(pml_y_size)]]])
hf.create_dataset('pml_z_size', data=[[[np.uint64(pml_z_size)]]])
hf.create_dataset('pml_x_alpha', data=[[[np.float32(pml_x_alpha)]]])
hf.create_dataset('pml_y_alpha', data=[[[np.float32(pml_y_alpha)]]])
hf.create_dataset('pml_z_alpha', data=[[[np.float32(pml_z_alpha)]]])

hf.create_dataset('pml_x', data=np.float32(pml_x.reshape(1,1,-1)))
hf.create_dataset('pml_x_sgx', data=np.float32(pml_x_sgx.reshape(1,1,-1)))
hf.create_dataset('pml_y', data=np.float32(pml_y.reshape(1,-1,1)))
hf.create_dataset('pml_y_sgy', data=np.float32(pml_y_sgy.reshape(1,-1,1)))
hf.create_dataset('pml_z', data=np.float32(pml_z.reshape(-1,1,1)))
hf.create_dataset('pml_z_sgz', data=np.float32(pml_z_sgz.reshape(-1,1,1)))





hf.close()





def save_h5(hf,var_name,var_value,castTo):
    if castTo == 'int':
        hf.create_dataset(var_name, data=np.uint64(var_value))
        hf[var_name].attrs['data_type'] = b'long'
    elif castTo == 'float':
        hf.create_dataset(var_name, data=np.float32(var_value))
        hf[var_name].attrs['data_type'] = b'float'
    else:
        print('Inadmissable data type')
        return
    
    hf[var_name].attrs['domain_type'] = b'real'