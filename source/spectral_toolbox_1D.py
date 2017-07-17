#!/usr/bin/env/ python3
"""
Objects for implementing spectral methods.

This module provides several methods contained within the SpectralToolbox class
for performing standard vector calculus operations in spectral space. It is
intended for use on a two-dimensional domain of equally-spaced gridpoints.

Classes
-------
- `SpectralToolbox` -- Superclass to implement standard techniques of spectral methods

Miscellaneous Functions for Unit Testing
----------------------------------------
- `test_func` -- Test function for unit testing
- `check_derivatives`
- `check_jacobian`
- `check_multiply`
- `check_laplacian`

All of the above compare the computed solution to an analytical solution.

Notes
-----
1) Invoking this module from the command line without any arguments will perform unit tests on the methods contained within.

2) The Fourier representation used is:

.. math:: u(x, y, t) = \\sum_{k_{1} = -N/2}^{N/2 -1} \\sum_{k_{2} = -N/2}^{N/2 - 1} \hat{u}(t) \\exp((2i\\pi/L)(k_{1}x + k_{2}y))

The spectrum is stored in the logical-to-mathematicians order, i.e. the wavenumbers are stored as:

+--------------------+--------------------+--------------------+--------------------+
|    (-N/2)(-N/2)    |  (-N/2)(-N/2 + 1)  |         ...        |   (-N/2)(N/2 - 1)  |
+--------------------+--------------------+--------------------+--------------------+
|  (-N/2 + 1)(-N/2)  |(-N/2 + 1)(-N/2 + 1)|         ...        |  (-N/2 + 1)(N/2 -1)|
+--------------------+--------------------+--------------------+--------------------+
|         ...        |         ...        |         ...        |         ...        |
+--------------------+--------------------+--------------------+--------------------+
|  (N/2 - 1)(-N/2)   | (N/2 - 1)(-N/2 + 1)|         ...        |  (N/2 - 1)(N/2 -1) |
+--------------------+--------------------+--------------------+--------------------+

With the Nyquist frequencies (corresponding to the most negative frequencies) located where one or both index = -N/2,
and the constant mode corresponding to :math:`k_{1} = 0`, :math:`k_{2} = 0` in the (N, N) position in the array.

2) The sign matrix, `sign_mat`, is used to relate the truncated Fourier series of a function, with wavenumbers that
run from [-N/2, N/2), with the FFT, with wavenumbers in [0, N). The point of doing this is to make the
operations in spectral space more intuitive from a mathematical perspective. We proceed by discretising the
desired function on an equispaced grid as above, noting that the sum runs from k = -N/2 to N/2 -1. In order to
make the sum correspond with the FFT, we perform the change of variables :math:`p = k + N/2`:

.. math:: f(x_{j}) = \\sum_{p=0}^{N-1} \\hat{f}(p-N/2) e^{2i\\pi k p/N}

.. math:: f(x_{j}) = \\sum_{p=0}^{N-1} \\hat{f}(p-N/2) (-1)^{p} e^{2i\\pi k/N}

Note that a factor of :math:`(-1)^{p}` is obtained. These alternating signs are implemented in the sign matrix
to permit a single multiplication operation.

3) The code will attempt to use the vastly superior FFTW routines if the user has the appropriate FFTW and PyFFTW
libraries installed. If not, it falls back to NumPy's native implementation.

See also
--------
numpy.fft, numpy, FFTW, PyFFTW

| Author: Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import numpy as np
#import matplotlib.pyplot as plt
try:
    import pyfftw
    fftw_flag = True
except ImportError:
    fftw_flag = False

class SpectralToolbox:
    """
    Implements the required vector calculus techniques for spectral methods.

    This class stores the required constants as attributes to be used by its
    methods for spectral methods in 2 spatial dimensions.

    **Attributes**

    - `N` : The number of grid points in the domain, such that there are N X N equally-spaced points
    - `L` : The side length of the domain
    - `factor` : Factor arising in spectral differentiation on non-2pi domain (see below)
    - `sign_mat` : Matrix to reorder spectral space (see below)
    - `deriv_mat_x1`, deriv_mat_x2` : matrices for spectral differentiation in x1 and x2 directions (see calc_derivative method)
    - `laplace_op` : Array to implement Laplacian (see laplacian method)

    **Methods**

    - `dealias_pad` -- Pads the input array with zeros to prevent aliasing
    - `dealias_unpad` -- Removes the padding from dealias_pad
    - `forward_fft` -- Wrapper to perform Fast Fourier Transform with desired normalisation
    - `inverse_fft` -- Wrapper to invert FFT with desired normalisation
    - `calc_derivative` -- Computes 1st or 2nd order derivatives along x and/or y directions
    - `multiply_nonlinear` -- Multiplies two functions in a pseudo-spectral fashion
    - `solve_inverse_laplacian` -- Solves the inverse Laplacian problem with a chosen constant of integration
    - `jacobian1` -- Computes Jacobian of a single function
    - `jacobian` -- Computes Jacobian of two functions
    - `laplacian` -- Computes the Laplacian of a given function

    **Example**

    | ``>> st = SpectralToolbox(128, 2*np.pi)``
    | ``>> du_dx = st.calc_derivative(A, 'x')  # Compute x-derivative``
    """

    def __init__(self, N, L, padFlag = True):
        self.fftw_flag = fftw_flag
        if fftw_flag:
            pyfftw.interfaces.cache.enable()
            self.fft_use = pyfftw.interfaces.numpy_fft
        else:
            self.fft_use = np.fft

        self.N = N
        if N%2: # Methods are not currently implemented for odd values of N
            raise ValueError("N must be even")
        self.L = L
        self.factor = 2.0*np.pi/L

        self.pad = padFlag

        # Create sign matrix
        self.sign_mat = np.zeros((2*self.N), dtype='complex')
        for k in range(0,2*self.N):
            self.sign_mat[k] = (-1)**(k)

        self.sign_mat_fft_big = self.sign_mat/(2.0*N)
        self.sign_mat_fft_small = self.sign_mat[0:N]/N

        self.sign_mat_ifft_big = (2.0*N)*self.sign_mat
        self.sign_mat_ifft_small = N*self.sign_mat[0:N]

        self.deriv_mat_x1 = np.zeros((self.N), dtype = complex)
        for k in range(-self.N//2,self.N//2):
            self.deriv_mat_x1[k+self.N//2] = 1j * k * self.factor

    def dealias_pad(self, A):
        """
        Pads the spectrum with zeros in high frequencies for aliasing control.

        This method doubles the size of the spectrum and sets all high frequencies to zero
        for aliasing control. This causes the resolution in real space (after inverse FFT)
        to be doubled. The higher frequencies are then discarded upon returning to
        spectral space. The array is doubled in a symmetric fashion.

        **Parameters**

        - `A` : the spectrum to be padded

        **Returns**

        - `B` : the padded spectrum

        **See also**

        dealias_unpad

        **Example**

        | ``>> A = np.ones((2,2), dtype = complex)``
        | ``>> st.dealias_pad(A)``
        | ``>> array([[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],``
        |          ``[ 0.+0.j,  1.+0.j,  1.+0.j,  0.+0.j],``
        |          ``[ 0.+0.j,  1.+0.j,  1.+0.j,  0.+0.j],``
        |          ``[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]])``

        """
        m = A.size
        B = np.zeros((2*m), dtype = complex)
        B[m//2:3*m//2] = A[:]
        return B

    def dealias_unpad(self, A):
        """
        Unpads the spectrum, removing high frequencies, for aliasing control.

        This method reverses the padding set up in the dealias_pad method,
        returning the size of the spectrum to the initial value upon returning
        to spectral space. The highest half of the frequencies are discarded.

        **Parameters**

        - `A` : the spectrum to be unpadded

        **Returns**

        - `B` : the unpadded spectrum

        **See also**

        dealias_pad

        **Example**

        | ``>> A = array([[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],``
        |            ``[ 0.+0.j,  1.+0.j,  1.+0.j,  0.+0.j],``
        |            ``[ 0.+0.j,  1.+0.j,  1.+0.j,  0.+0.j],``
        |            ``[ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]])``
        | ``>> st.dealias_unpad(A)``
        | ``>> array([[ 1.+0.j,  1.+0.j],``
        |        ``[ 1.+0.j,  1.+0.j]])``
        """
        m = A.size
        B = np.zeros((m//2), dtype = complex)
        B = A[m//4:3*m//4]
        return B

    def forward_fft(self, array_in):
        """
        Wrapper to implement the forward Fast Fourier Transform.

        This method implements the forward FFT (i.e. from real-space
        to spectral space). The values are normalised such that the spectral
        coefficient corresponds to a wave of exactly that amplitude, rather
        than the normalised value returned by the standard FFT. The sign matrix
        is used to yield a spectrum running from -N/2 to N/2 -1.

        **Parameters**

        - `array_in` : the array of real values

        **Returns**

        - `out` : the spectral coefficients, in the chosen framework

        **See Also**

        Note 2 in the class header (`sign matrix`), inverse_fft

        """
        # Find side length, as real array may or may not be doubled for
        # aliasing control
        side = array_in.size

        if side == self.N:
            out = self.fft_use.fft(self.sign_mat_fft_small*array_in)
        elif side == 2*self.N:
            out = self.fft_use.fft(self.sign_mat_fft_big*array_in)
        return out

    def inverse_fft(self, array_in):
        """
        Wrapper to implement the inverse Fast Fourier Transform.

        This method implements the inverse FFT (i.e. from spectral space
        to real space). The values are normalised such that the spectral
        coefficient corresponds to a wave of exactly that amplitude, rather
        than the normalised value returned by the standard FFT. The sign matrix
        is used to yield a spectrum running from -N/2 to N/2 -1.

        **Parameters**

        - `array_in` : the array of spectral coefficients

        **Returns**

        - `out` : the real values at gridpoints, in the chosen framework

        **See Also**

        Note 2 in the class header (sign matrix), forward_fft

        """
        # Find side length, as spectrum may or may not have been doubled
        # for aliasing control
        side = array_in.size

        if side == self.N:
            out = self.sign_mat_ifft_small*self.fft_use.ifft(array_in)
        elif side == 2*self.N:
            out = self.sign_mat_ifft_big*self.fft_use.ifft(array_in)
        return out

    def calc_derivative(self, array_in):
        """
        Computes 1st or 2nd order derivatives in x and/or y directions.

        This method implements spectral (spatial) differentiation through the use of
        the derivative matrices, stored as attributes of the SpectralToolbox class.
        The Nyquist frequency is treated specially for odd orders of multiplication.

        **Parameters**

        - `array_in` : the array of spectral coefficients to be differentiated
        - `direction1` : the first direction to be differentiated ('x' or 'y')
        - `direction2` : the optional second direction to be differentiated ('x' or 'y')

        **Returns**

        - `out` : the differentiated spectrum

        **Notes**

        Recall that the function, u, is represented in Fourier space as:

        .. math:: u(x, y, t) = \\sum_{k_{1} = -N/2}^{N/2 -1} \\sum_{k_{2} = -N/2}^{N/2 - 1} \\hat{u}(t) \\exp((2i\\pi/L)(k_{1}x + k_{2}y))

        Then the first derivative in the x-direction, as :math:`\\hat{u}(t)` is not a function of space, is simply:

        .. math:: \\frac{\\partial u(x, y, t)}{\\partial x} = \\sum_{k_{1} = -N/2}^{N/2 -1} \\sum_{k_{2} = -N/2}^{N/2 - 1} \\hat{u}(t)  (2i\\pi k_{1}/L)  \\exp((2i\\pi/L)(k_{1}x + k_{2}y))

        Thus, differentiation in spectral space is a matter of multiplying each Fourier mode by the length factor (:math:`B=2\\pi/L`) times i times the associated wavenumber.
        To reduce loops and make the implementation simpler, these modes are contained in the derivatrive matrices, which take the form:

        .. math:: \\text{deriv_mat_x1} = \\left[\\begin{array}{cccc}
                        iBk_{1} & iBk_{1} & \\cdots & iBk_{1} \\\\
                       iBk_{2} & iBk_{2} & \\cdots & iBk_{2}\\\\
                       \\vdots & \\vdots & \\ddots & \\vdots \\\\
                       iBk_{n} & iBk_{n} & \\cdots & iBk_{n} \\end{array}\\right]

        and:

        .. math:: \\text{deriv_mat_x2} = \\left[\\begin{array}{cccc}
                        iBk_{1} & iBk_{2} & \\cdots & iBk_{n} \\\\
                       iBk_{1} & iBk_{2} & \\cdots & iBk_{n}\\\\
                       \\vdots & \\vdots & \\ddots & \\vdots \\\\
                       iBk_{1} & iBk_{2} & \\cdots & iBk_{n} \\end{array}\\right]

        **Example**

        | ``>> A = array([[ 1.+0.j,  1.+0.j],``
        |           ``[ 1.+0.j,  1.+0.j]])``
        |
        | ``>> st.calc_derivative(A, 'x')``
        | ``>> array([[ 0.+0.j,  0.+0.j],``
        |        ``[ 0.+0.j,  0.+0.j]])``

        """
        A = array_in.copy()
        # Perform first derivative in desired direction
        out = self.deriv_mat_x1*A

        return out

    def multiply_nonlinear(self, array1, array2):
        """
        Simple method for multiplying two sets of data when nonlinearities are involved
        and transformation into real space is necessary.

        This method computes the product of two quantities defined in spectral space
        using a pseudo-spectral method, i.e. the multiplication is performed in realspace.
        The spectra are padded for aliasing control and must be of the same dimension.

        **Parameters**

        - `array_1`, `array_2` : the input spectra to be multiplied

        **Returns**

        - `f3_hat` : the spectrum obtainined from the multiplication

        **See Also**

        dealias_pad, dealias_unpad

        """
        if self.pad:
            # compute grid values via FFT
            f1_vals = self.inverse_fft(self.dealias_pad(array1))
            f2_vals = self.inverse_fft(self.dealias_pad(array2))

            # multiply in space
            f3_vals = f1_vals * f2_vals

            # compute Fourier coeffs
            f3_hat = self.dealias_unpad(self.forward_fft(f3_vals))
        else:
            f1_vals = self.inverse_fft(self.dealias_unpad(array1))
            f2_vals = self.inverse_fft(self.dealias_unpad(array2))
            f3_vals = f1_vals * f2_vals
            f3_hat = self.forward_fft(f3_vals)

        return f3_hat

#Unit Tests Below#

def testfunc(x):

    return np.sin(2.0*x)*np.cos(3.0*x) + np.cos(8.0*x)*np.sin(11.0*x)

def check_derivatives(st, N, L):
    """

    """
    u = np.zeros((N),dtype=complex)
    u_analytical_x = np.zeros((N))
    sp = np.zeros(N)

    for n1 in range(0,N):
        x = L*n1/N
        sp[n1] = x
        u[n1] = testfunc(x)
        u_analytical_x[n1] = 2.0*np.cos(2.0*x)*np.cos(3.0*x) - 3.0*np.sin(2.0*x)*np.sin(3.0*x) - 8.0*np.sin(8.0*x)*np.sin(11.0*x) + 11.0*np.cos(8.0*x)*np.cos(11.0*x)

    u_hat = st.forward_fft(u)
    u_hat_x = st.calc_derivative(u_hat)
    u_num_x = st.inverse_fft(u_hat_x)

    plt.plot(sp,u_num_x,sp,u_analytical_x)
    plt.show()
    return(np.max(np.absolute(u_num_x - u_analytical_x)))

def check_multiply(st, N, L):
    """

    """
    u1 = np.zeros((N),dtype=complex)
    u2 = np.zeros((N),dtype=complex)
    u_analytical = np.zeros((N))

    for n1 in range(0,N):
            x = L*n1/N
            u1[n1] = np.sin(2.0*x)*np.cos(4.0*x)
            u2[n1] = np.sin(3.0*x)*np.cos(3.0*x)

    u_analytical = u1*u2
    u_hat1 = st.forward_fft(u1)
    u_hat2 = st.forward_fft(u2)
    u_hat = st.multiply_nonlinear(u_hat1, u_hat2)
    u_num = st.inverse_fft(u_hat)

    return(np.max(np.absolute(u_num - u_analytical)))#/(u_analytical)


if __name__ == "__main__":

    np.set_printoptions(linewidth = 140, precision = 2)

    N = 512
    L = 2.0*np.pi
    st = SpectralToolbox(N, L, padFlag = True)
    out = check_derivatives(st, N, L)
    print("Error in x-derivative is: {}".format(out))
    out = check_multiply(st, N, L)
    print("Error in Multiply is: {}".format(out))

