# This script computes the Helmoltz decomposition of a 3-D vector field Psi
# INPUT: the (x,y,z) components of the vector field 
# OUTPUT: the (x,y,z) components of the divergence-free and curl-free vector fields
#
# By default, the divergence-free and curl-free vectors are written to files
# THe default file formats are Numpy binary (.npy) and unformatted binary (.dat).
# In both cases, the code assumes the input array are flattened and reshape them according to the saphe of the array(nygrid,nygrid,nzgrid)
# The array ordering is by defaults 'C', but can be changed.
#
# DISCLAIMER: tested only for cubic grids, i.e. for nxgrid=nygrid=nzgrid
# **********************************************                                                                                                                                                     
# **********************************************                                                                                                                                                     
# **********************************************  
# INPUT PARAMETERS

nxgrid = 256
nygrid = 256
nzgrid = 256

lxbox = 200
lybox = 200
lzbox = 200

lxcell = lxbox / nxgrid
lycell = lybox / nygrid
lzcell = lzbox / nzgrid

input_directory = ''
output_directory = ''

psix_in_filename = 'psix.npy'
psiy_in_filename = 'psiy.npy'
psiz_in_filename = 'psiz.npy'

psix_out_divfree_filename = 'psix_divfree.npy'
psiy_out_divfree_filename = 'psiy_divfree.npy'
psiz_out_divfree_filename = 'psiz_divfree.pyy'

psix_out_curlfree_filename = 'psix_curlfree.npy'
psiy_out_curlfree_filename = 'psiy_curlfree.npy'
psiz_out_curlfree_filename = 'psiz_curlfree.npy'

file_format = 'np_binary' # 'np_binary' or 'u_binary'
prec = 'float32' # relevant only for 'u_binary'. 'np_binary' will find out alone

order ='C'

# **********************************************                                                                                                                                                     
# **********************************************                                                                                                                                                     
# **********************************************  
# MAIN

import numpy as np

if file_format == 'u_binary':
    if prec=='float32':
        dtype=np.float32
    elif prec=='float64':
        dtype=np.float64
    else:
        print('Error: precision type not found. Exiting.')
        exit()

    psix = np.fromfile(input_directory+psix_in_filename, dtype=dtype)
    psiy = np.fromfile(input_directory+psiy_in_filename, dtype=dtype)
    psiz = np.fromfile(input_directory+psiz_in_filename, dtype=dtype)

elif file_format == 'np_binary':

    psix = np.load(input_directory+psix_in_filename)
    psiy = np.load(input_directory+psiy_in_filename)
    psiz = np.load(input_directory+psiz_in_filename)

else:
    print('Error: file format not found. Exiting.')
    exit()

psix = np.reshape(psix, (nxgrid,nygrid,nzgrid), order=order)
psiy = np.reshape(psiy, (nxgrid,nygrid,nzgrid), order=order)
psiz = np.reshape(psiz, (nxgrid,nygrid,nzgrid), order=order)
    
psix_f = np.fft.fftn(psix)
psiy_f = np.fft.fftn(psiy)
psiz_f = np.fft.fftn(psiz)

kx = np.fft.fftfreq(nxgrid).reshape(nxgrid,1,1)
ky = np.fft.fftfreq(nxgrid).reshape(nxgrid,1)
kz = np.fft.fftfreq(nxgrid)

k2 = kx**2 + ky**2 + kz**2
k2[0,0,0] = 1. # to avoid inf

divpsi_f = (psix_f * kx +  psiy_f * ky + psiz_f * kz) #* 1j

divpsi_f_overk = divpsi_f / k2

# Back to real space
psi_curlfree_x = np.fft.ifftn(divpsi_f_overk * kx).real 
psi_curlfree_y = np.fft.ifftn(divpsi_f_overk * ky).real
psi_curlfree_z = np.fft.ifftn(divpsi_f_overk * kz).real

psi_divfree_x = psix - psi_curlfree_x
psi_divfree_y = psiy - psi_curlfree_y
psi_divfree_z = psiz - psi_curlfree_z

psi_divfree_x = psi_divfree_x.flatten(order=order)
psi_divfree_y = psi_divfree_y.flatten(order=order)
psi_divfree_z = psi_divfree_z.flatten(order=order)

psi_curlfree_x = psi_curlfree_x.flatten(order=order)
psi_curlfree_y = psi_curlfree_y.flatten(order=order)
psi_curlfree_z = psi_curlfree_z.flatten(order=order)

# Write fields to file

if file_format == 'u_binary':

    psi_divfree_x.astype(prec).tofile(output_directory+psix_out_divfree_filename)
    psi_divfree_y.astype(prec).tofile(output_directory+psiy_out_divfree_filename)
    psi_divfree_z.astype(prec).tofile(output_directory+psiz_out_divfree_filename)

    psi_curlfree_x.astype(prec).tofile(output_directory+psix_out_curlfree_filename)
    psi_curlfree_y.astype(prec).tofile(output_directory+psiy_out_curlfree_filename)
    psi_curlfree_z.astype(prec).tofile(output_directory+psiz_out_curlfree_filename)

elif file_format == 'np_binary':

    np.save(output_directory+psix_out_divfree_filename, psi_divfree_x)
    np.save(output_directory+psiy_out_divfree_filename, psi_divfree_y)
    np.save(output_directory+psiz_out_divfree_filename, psi_divfree_z)

    np.save(output_directory+psix_out_curlfree_filename, psi_curlfree_x)
    np.save(output_directory+psiy_out_curlfree_filename, psi_curlfree_y)
    np.save(output_directory+psiz_out_curlfree_filename, psi_curlfree_z)
  
else:

    print('Error: file forat not found. Exiting')
    exit()