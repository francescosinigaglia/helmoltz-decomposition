# helmoltz-decomposition
Script to perform Helmoltz decomposition of a 3-D vector field in a cubic box.

The script is coded in Python and uses requires only the NumPy package.

You will just have to set as input the grid information (Nx, Ny, Nz), the size of the box, and the I/O file names.

It works with NumPy binary format (.npy) and unformatted binary (.dat). In both cases, the code assumes the input array are flattened and reshape them to 3-D. The array ordering is by defaults 'C', but can be changed.


