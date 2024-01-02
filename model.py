import numpy as np

def conv(M, F, S):
    Iw = M.shape[1]
    Ih = M.shape[0]
    
    Fw = F.shape[1]
    Fh = F.shape[0]
    
    Ow = int((Iw-Fw)/S) + 1
    Oh = int((Ih-Oh)/S) + 1
    
    matrix_result = np.zeros((h, w))
    
    for i in range(matrix_result.shape[0]):
        for j in range(matrix_result.shape[1]):
            bloc = matrix_result[(i*S):(i*S)+Fh, (i*S):(i*S)+Fw]