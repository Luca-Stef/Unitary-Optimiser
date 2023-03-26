"""

Reimplementation of Jamie's staircase decomposition algorithm.

See `ml_mera/ml_mera_lewis/code/jamie_initalise.py`

"""

import numpy as np
from scipy.linalg import polar
from functools import reduce
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import unitary_group
from cirq import two_qubit_matrix_to_ion_operations


I = np.identity(2)

def two_qubit_unitary_to_circuit(circuit, U, q0, q1):
    '''
    Decompose the two qubit unitary U into a sequence of gates acting on qubits
    q0 and q1 in the circuit.
    '''
    ops = two_qubit_matrix_to_ion_operations(q0=q0, q1=q1, mat=U,
                                         allow_partial_czs=False)

    circuit.append(ops)

def staircase_decompose(U, first='l', depth=1, α=1e-1, tol=1e-10, maxiters=1000, α_min=None, α_steps=None):
    '''
    Perform a decomposition of an arbitrary unitary U to a staircase circuit.

    Credit: Fergus Barratt
    '''
    if α_min is not None:
        if α_steps is None:
            α_steps = 10
        α_range = np.linspace(α, α_min, α_steps)
        α_ind = 0
    n_qubits = int(np.log2(U.shape[0]))
    Us0 = [[np.linalg.qr(np.random.randn(4, 4)+1j*np.random.randn(4, 4))[0] for _ in range(n_qubits-1)] for _ in range(depth)]
    def overlap(Us):
        overlap = U.conj().T
        for row in Us:
            for i, U2 in enumerate(row):
                if first=='l':
                    overlap = overlap @ reduce(np.kron, [I]*i+[U2] +[I]*(n_qubits-i-2))
                elif first=='r':
                    overlap = overlap @ reduce(np.kron, [I]*(n_qubits-i-2)+[U2] +[I]*(i))
        return np.trace(overlap)/U.shape[0]

    def dEdU(Us, j, k, testing=True):
        ''' differentiate wrt. jth U, in kth layer'''
        #print('Updating depth {}, row {}'.format(k, j))
        pre = U.conj().T
        post = reduce(np.kron, [I]*n_qubits)
        for l, row in enumerate(Us):
            if l==k:
                if first=='r':
                    #print('pre: ')
                    for i, U2 in enumerate(row[:j]):
                        #print('I'*(n_qubits-i-2) + 'UU'+'I'*(i))
                        pre = pre @ reduce(np.kron, [I]*(n_qubits-i-2)+[U2] +[I]*(i))

                    #print('post: ')
                    for i, U2 in enumerate(row[j+1:]):
                        i = j+i+1
                        #print('I'*(n_qubits-i-2) + 'UU'+'I'*(i))
                        post = post @ reduce(np.kron, [I]*(n_qubits-i-2)+[U2] +[I]*(i))

                if first=='l':
                    #print('pre: ')
                    for i, U2 in enumerate(row[:j]):
                        #print('I'*i + 'UU'+'I'*(n_qubits-i-2))
                        pre = pre @ reduce(np.kron, [I]*i+[U2] +[I]*(n_qubits-i-2))

                    #print('post: ')
                    for i, U2 in enumerate(row[j+1:]):
                        i = j+i+1
                        #print('I'*i + 'UU'+'I'*(n_qubits-i-2))
                        post = post @ reduce(np.kron, [I]*i+[U2] +[I]*(n_qubits-i-2))
            elif l<k:
                #print('pre!: ')
                for i, U2 in enumerate(row):
                    if first=='l':
                        #print('I'*i + 'UU'+'I'*(n_qubits-i-2))
                        pre = pre @ reduce(np.kron, [I]*i+[U2] +[I]*(n_qubits-i-2))
                    elif first=='r':
                        #print('I'*(n_qubits-i-2) + 'UU'+'I'*(i))
                        pre = pre @ reduce(np.kron, [I]*(n_qubits-i-2)+[U2] +[I]*(i))
            elif l>k:
                #print('post!: ')
                for i, U2 in enumerate(row):
                    if first=='l':
                        #print('I'*i + 'UU'+'I'*(n_qubits-i-2))
                        post = post @ reduce(np.kron, [I]*i+[U2] +[I]*(n_qubits-i-2))
                    elif first=='r':
                        #print('I'*(n_qubits-i-2) + 'UU'+'I'*(i))
                        post = post @ reduce(np.kron, [I]*(n_qubits-i-2)+[U2] +[I]*(i))
#        if first=='l':
#            print('Updating row ({},{})'.format(j, j+1))
#        else:
#            print('Updating row ({}, {})'.format(n_qubits - j - 2, n_qubits - j - 1))

        pre = pre.reshape([2]*(2*n_qubits))
        post = post.reshape([2]*(2*n_qubits))

        if first=='l':
            td_links = [list(range(n_qubits+j)) + list(range(n_qubits+j+2, 2*n_qubits))] + [list(range(n_qubits, 2*n_qubits))+ list(range(j))+list(range(j+2, n_qubits))]
        else:
            td_links = [list(range(2*n_qubits-j-2)) + list(range(2*n_qubits-j, 2*n_qubits))] + [list(range(n_qubits, 2*n_qubits))+ list(range(n_qubits-j-2))+list(range(n_qubits-j, n_qubits))]
#        print(td_links)
#        print('')
#        print('\n')
        dEdU = np.tensordot(pre, post, td_links).reshape((4, 4)).conj()
        return dEdU

    Us = np.array(Us0)
    e0 = np.abs(overlap(Us))**2
    es = [e0]
    for i in tqdm(range(maxiters)):
        for j in range(n_qubits-1):
            for k in range(depth):
                Us[k, j] = polar(Us[k, j]+α*dEdU(Us, j, k, testing=False))[0]
        es.append(np.abs(overlap(Us))**2)
        dE = es[-1]-es[-2]
        if np.abs(dE)<tol:
            if α_min is not None:
                if α_ind == len(α_range) - 1:
                    break
                α_ind += 1
                α = α_range[α_ind]
                print("Changed α: {}".format(α))
            else:
                break
    return Us, np.array(es)

if __name__=="__main__":

    m_iters = 500
#    α = 5e-2
    depth=8
    n_qubits = 4
    U = unitary_group.rvs(2**4)

    Usr, esr = staircase_decompose(U, first='r', depth=depth, maxiters=m_iters)
    Usl, esl = staircase_decompose(U, first='l', depth=depth, maxiters=m_iters)

    fig = plt.figure()
    plt.plot(esl, label='1e-5 with changin α')

    plt.legend()
    plt.title('Fergus decomp left')

    fig = plt.figure()
    plt.plot(esr, label='1e-5 with changin α')

    plt.legend()
    plt.title('Fergus decomp right')
    plt.show()
