import numpy as np
import interferometer as itf
from tools.decomposition import decomposition, Fredkin

X_input = Fredkin

A = decomposition(
    X_input, tentativas=200, method='trf')

# SVD decomposition
Um, Sm, Vtm = np.linalg.svd(A, full_matrices=True)

# construir sigma atenuada
Sm_att = np.zeros((6, 6))
for i in range(len(Sm)):
    Sm_att[i, i] = (Sm[i] / Sm[0])

# obtenção dos circuitos
U = itf.square_decomposition(Um)
S_att = itf.square_decomposition(Sm_att)
Vt = itf.square_decomposition(Vtm)

# Printar circutios individuais
U.draw()
S_att.draw()
Vt.draw()