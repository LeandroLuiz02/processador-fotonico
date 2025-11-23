import logging
import numpy as np

def svd_decomposition(A: np.array):
    """
    Function to execute the SVD decomposition in Python
    """
    logging.debug("--- 1. Original Matrix A ---")
    logging.debug(A)
    logging.debug(f"Input Matrix Shape: {A.shape}")

    U, S, Vt = np.linalg.svd(A, full_matrices=True)

    logging.debug("--- 2. Decomposition Components ---")
    
    logging.debug("Matrix U (Left Singular Vectors):")
    logging.debug(U)
    logging.debug(f"Shape: {U.shape}\n")

    
    S_final = np.zeros(A.shape)
    for i in range(A.shape[0]):
        S_final[i,i] = S[i]

    logging.debug("Vector S (Singular Values):")
    logging.debug(S_final)
    logging.debug(f"Shape: {S_final.shape}\n")

    logging.debug("Matrix Vt (Right Singular Vectors Transposed):")
    logging.debug(Vt)
    logging.debug(f"Shape: {Vt.shape}\n")

    return U, S_final, Vt