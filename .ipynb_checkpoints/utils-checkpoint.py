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

def print_circuit_structure(phis, thetas, alphas):
    rows, cols = phis.shape
    dim = rows + 1 # Number of waveguides (N)

    print(f"--- CIRCUIT CONSTRUCTION (N={dim}) ---\n")

    # 1. Iterate through layers (Columns of the matrices)
    for p in range(cols):
        print(f"Layer {p}:")
        
        # Determine if this is an even or odd layer
        # If p is even, we connect (0,1), (2,3)...
        # If p is odd, we connect (1,2), (3,4)...
        
        start_row = 0 
        
        # NOTE: The reconstruct code creates a specific checkerboard.
        # It usually alternates. Let's look at the provided 'reconstruct' logic:
        # It runs `range(0, row, 2)` THEN `range(1, row, 2)` for EVERY column.
        # However, standard Clements alternates. 
        # Let's verify the standard Clements reconstruction logic from the snippet:
        # The snippet applies BOTH even and odd pairs in ONE column loop if possible, 
        # but standard Clements spreads them out.
        
        # Let's stick to the indices strictly present in the matrices.
        # If phis[q, p] is non-zero (or relevant), there is a block.
        
        found_block = False
        for q in range(rows):
            # We check if this q,p coordinate was actually filled/used.
            # In a full rectangular mesh, usually they are all used unless N is small.
            
            # Logic based on standard Clements Rectangular Mesh:
            is_even_layer = (p % 2 == 0)
            is_even_row   = (q % 2 == 0)
            
            # Ideally, we place blocks based on the 'checkerboard'
            if (is_even_layer and is_even_row) or (not is_even_layer and not is_even_row):
                phi_val = phis[q, p]
                theta_val = thetas[q, p]
                print(f"  [MZI] connecting WG-{q} & WG-{q+1} | phi={phi_val:.3f}, theta={theta_val:.3f}")
                found_block = True
        
        if not found_block:
            print("  (No blocks in this layer - check matrix sparsity)")
        print("")

    # 2. Output Alphas
    print("Output Phase Screen:")
    for i, alpha in enumerate(alphas):
        print(f"  WG-{i}: Phase Shifter = {alpha:.3f}")