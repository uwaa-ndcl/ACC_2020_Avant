import numpy as np

def print_matrix_as_latex(mat, n_digs=3):
    '''
    print a matrix in latex form
    inputs:
        mat: matrix, n_digs: number of digits in largest value
    '''
    n_rows, n_cols = mat.shape
    mat_max = np.amax(mat) # max value in matrix
    max_pow = int(np.floor(np.log10(mat_max))) # max value power of 10
    # power of 10 to divide matrix by so largest element is in range [100,999]
    scl_pow = max_pow - (n_digs - 1)
    scl = 10**scl_pow # value to divide matrix by
    mat_scl = mat/scl

    # print in latex form
    print('10^{' + str(scl_pow) + '} \\times')
    print(r'\begin{bmatrix}')
    for i in range(n_rows):
        for j in range(n_cols):
            # not the last column
            if j < n_cols - 1:
                print('%.0f' %  mat_scl[i, j], '& ', end='')

            # last column
            else:    
                # not last row
                if i < n_rows - 1:
                    print('%.0f' %  mat_scl[i, j], '\\\ ')

                # last column and last row
                else:
                    print('%.0f' %  mat_scl[i, j])

    print(r'\end{bmatrix}')
