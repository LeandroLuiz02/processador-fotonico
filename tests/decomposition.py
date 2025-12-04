import numpy as np
from scipy.optimize import least_squares
import itertools as it

def resolver_matriz_A(X_target, n=6, pares=None, tentativas=200, chute_scale=1.0, method='trf', nonneg=True):
    """
    Recupera uma matriz A (n x n) tal que, para os TRIPLOS em `pares`,
    a matriz X calculada via PERMANENTE de submatriz 3x3 se aproxime de X_target.

    Observação: agora cada equação usa uma submatriz 3x3 extraída de A,
    com linhas (r1,r2,r3) e colunas (c1,c2,c3). A permanente 3x3 soma
    todos os produtos de A[r_k, c_{perm(k)}] para permutações de 3 elementos,
    sem sinais alternados.
    """
    import numpy as np
    from scipy.optimize import least_squares
    import itertools as it

    def permanent_3x3(M):
        # M é array 3x3; permanente = soma sobre todas as 3! permutações
        perms = it.permutations([0,1,2])
        s = 0.0
        for p in perms:
            s += M[0, p[0]] * M[1, p[1]] * M[2, p[2]]
        return s

    if pares is None:

        pares = [
            (0, 2, 4),
            (0, 2, 5),
            (0, 3, 4),
            (0, 3, 5),
            (1, 2, 4),
            (1, 2, 5),
            (1, 3, 4),
            (1, 3, 5),
        ]

    m = len(pares)
    # assert X_target.shape == (m, m), f"X_target deve ser {m}x{m}, recebido {X_target.shape}"

    def residuos(vars_flat):
        A = vars_flat.reshape(n, n)
        diffs = []
        for i in range(m):
            r1, r2, r3 = pares[i]
            for j in range(m):
                c1, c2, c3 = pares[j]
                subM = np.array([
                    [A[r1, c1], A[r1, c2], A[r1, c3]],
                    [A[r2, c1], A[r2, c2], A[r2, c3]],
                    [A[r3, c1], A[r3, c2], A[r3, c3]],
                ])
                val_calc = permanent_3x3(subM)
                diffs.append(val_calc - X_target[i, j])
        return np.array(diffs)

    # print(f"Iniciando busca n={n}, {m} triplos (X {m}x{m}), até {tentativas} tentativas... (nonneg={nonneg})")

    melhor_erro = float('inf')
    melhor_solucao = None
    melhor_SVD = float('inf')

    lb = 0.0 if nonneg else -np.inf
    ub = np.inf

    for t in range(tentativas):
        chute = np.random.uniform(0 if nonneg else -chute_scale, chute_scale, n*n)
        if nonneg:
            chute = np.clip(chute, 0, None)
        resultado = least_squares(
            residuos,
            chute,
            method=method,
            bounds=((-np.inf*np.ones(n*n), np.inf*np.ones(n*n)))
        )
        erro_atual = resultado.cost

        # TODO: retornar matriz com menor valor singular S[0]

        if erro_atual < 1e-12:
            print(f"-> Solução encontrada na tentativa {t+1}!")
            print(np.round(resultado.x.reshape(n, n), 5))
            U, S, Vt = np.linalg.svd(resultado.x.reshape(n, n), full_matrices=True)
            print(f"Maior valor singular de A_encontrada: {S[0]:.10f}")
            print(f"erro: {erro_atual}\n")

            if melhor_SVD > S[0]:
                melhor_SVD = S[0]
                melhor_solucao = resultado.x.reshape(n, n)

    if melhor_solucao is not None:
        print(f"A solução exata com menor maior valor singular S[0]: {melhor_SVD:.10f}")
        return melhor_solucao
    raise ValueError("Solução exata não encontrada em nenhuma tentativa.")

        

        #     # return resultado.x.reshape(n, n)
        # if erro_atual < melhor_erro:
        #     melhor_erro = erro_atual
        #     melhor_solucao = resultado.x.reshape(n, n)

    # print(f"Aviso: Solução exata não encontrada. Melhor erro residual: {melhor_erro}")
    # return melhor_solucao

def result_to_matrix(flat_arr, n):
    return flat_arr.reshape(n, n)

# --- TESTE (n=6, X 8x8, usando TRIPLOS e permanente 3x3) ---

# # 1. Matriz Real (A original) 6x6 não-negativa
# A_real = np.array([
#     [0.8, 0.1, 0.7, 0.3, 0.5, 0.2],
#     [0.6, 0.9, 0.2, 0.4, 0.3, 0.7],
#     [0.2, 0.5, 0.6, 0.1, 0.4, 0.8],
#     [0.7, 0.3, 0.5, 0.6, 0.2, 0.1],
#     [0.3, 0.4, 0.1, 0.2, 0.9, 0.5],
#     [0.5, 0.2, 0.8, 0.3, 0.6, 0.4]
# ], dtype=float)

n = 6
# TRIPLOS conforme especificado
pares = [
    (0, 2, 4),
    (0, 2, 5),
    (0, 3, 4),
    (0, 3, 5),
    (1, 2, 4),
    (1, 2, 5),
    (1, 3, 4),
    (1, 3, 5),
]

def permanent_3x3(M):
    s = 0.0
    for p in it.permutations([0,1,2]):
        s += M[0, p[0]] * M[1, p[1]] * M[2, p[2]]
    return s

m = len(pares)
X_input = np.zeros((m, m))
# for i in range(m):
#     r1, r2, r3 = pares[i]
#     for j in range(m):
#         c1, c2, c3 = pares[j]
#         subM = np.array([
#             [A_real[r1, c1], A_real[r1, c2], A_real[r1, c3]],
#             [A_real[r2, c1], A_real[r2, c2], A_real[r2, c3]],
#             [A_real[r3, c1], A_real[r3, c2], A_real[r3, c3]],
#         ])
#         X_input[i, j] = permanent_3x3(subM)

# print("Matriz X (Alvo) 8x8 (permanente 3x3):\n", np.round(X_input, decimals=3))
# print("Min/Max de X_input:", np.min(X_input), np.max(X_input))
# print("-" * 30)

Toffoli = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]])

Fredkin = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]])

X_input = Toffoli

A_encontrada = resolver_matriz_A(
    X_input, n=n, pares=pares, tentativas=5, chute_scale=1.0, method='trf', nonneg=False
)

# TODO: printar a matriz diagonal com os valores singulares de A_encontrada
U, S, Vt = np.linalg.svd(A_encontrada, full_matrices=True)
identidade_S = np.zeros((n, n))
for i in range(len(S)):
    identidade_S[i, i] = S[i]
print("Matriz de Valores Singulares S de A_encontrada:")
print(np.round(identidade_S, 5))


# print(np.round(A_encontrada, 5))

# 4. Validar o resultado recalculando X a partir da A encontrada (permanente 3x3)

# def permanent_3x3(M):
#     s = 0.0
#     for p in it.permutations([0,1,2]):
#         s += M[0, p[0]] * M[1, p[1]] * M[2, p[2]]
#     return s

# m = len(pares)
# X_check = np.zeros((m, m))
# for i in range(m):
#     r1, r2, r3 = pares[i]
#     for j in range(m):
#         c1, c2, c3 = pares[j]
#         subM = np.array([
#             [A_encontrada[r1, c1], A_encontrada[r1, c2], A_encontrada[r1, c3]],
#             [A_encontrada[r2, c1], A_encontrada[r2, c2], A_encontrada[r2, c3]],
#             [A_encontrada[r3, c1], A_encontrada[r3, c2], A_encontrada[r3, c3]],
#         ])
#         X_check[i, j] = permanent_3x3(subM)

# print("Min/Max de X_check:", np.round(np.min(X_check), 5), np.round(np.max(X_check), 5))
# erro_maximo = np.max(np.abs(X_input - X_check))
# print(f"\nErro Máximo (X_alvo - X_recalculado): {erro_maximo:.10f}")

# if erro_maximo < 1e-5:
#     print("SUCESSO: A matriz encontrada gera exatamente a matriz X (8x8).")
# else:
#     print("FALHA: A matriz encontrada não gera a matriz X.")

# print(np.round(X_check, 5))