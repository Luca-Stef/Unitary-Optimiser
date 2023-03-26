import numpy as np
from ncon import ncon

def rnd(*x):
    return np.random.randn(*x) + 1j*np.random.randn(*x)

def overlap(nodes): # needs revising

    mps = []
    rank = len(nodes)
    indices = []
    dummy = 1
    for i in range(rank):
        if i == op_site:
            mps += [nodes[op_site], o, nodes[op_site].conj()]
            indices += [dummy, dummy+2, dummy+4], [dummy+2, dummy+3], [dummy+1, dummy+3, dummy+5]
            dummy += 4
            continue
        mps += [nodes[i], nodes[i].conj()]
        if i == 0:
            indices += [dummy, dummy+1], [dummy, dummy+2]
            dummy += 1
            continue
        if i == rank-1:
            indices += [dummy, dummy+2], [dummy+1, dummy+2]
            break
        indices += [dummy, dummy+2, dummy+3], [dummy+1, dummy+2, dummy+4]
        dummy += 3
    
    return ncon(mps, indices)

def left_env(nodes, site):
    """
    Evironment to the left of site specified (sites numbered left to right starting from 1). 
    """
    mps = []
    indices = []
    dummy = 1
    for i in range(site-1):
        mps += [nodes[i], nodes[i].conj()]
        if site == 2:
            indices += [dummy, -1], [dummy, -2]
            break
        elif i == 0:
            indices += [dummy, dummy+1], [dummy, dummy+2]
            dummy += 1
            continue
        elif i == site-2:
            indices += [dummy, dummy+2, -1], [dummy+1, dummy+2, -2]
            break
        indices += [dummy, dummy+2, dummy+3], [dummy+1, dummy+2, dummy+4]
        dummy += 3

    return ncon(mps, indices)

def right_env(nodes, site):
    """
    Evironment to the left of site specified (sites numbered left to right starting from 1).
    """
    rank = len(nodes)
    mps = []
    indices = []
    dummy = 1
    for i in range(site, rank):
        mps += [nodes[i], nodes[i].conj()]
        if site == rank-1:
            indices += [-1, dummy], [-2, dummy]
            break
        if i == site:
            indices += [-1, dummy, dummy+1], [-2, dummy, dummy+2] 
            dummy += 1
            continue
        elif i == rank-1:
            indices += [dummy, dummy+2], [dummy+1, dummy+2]
            break
        indices += [dummy, dummy+2, dummy+3], [dummy+1, dummy+2, dummy+4]
        dummy += 3

    return ncon(mps, indices)

def norm(nodes):

    mps = []
    rank = len(nodes)
    indices = []
    dummy = 1
    for i in range(rank):

        mps += [nodes[i], nodes[i].conj()]
        if i == 0:
            indices += [dummy, dummy+1], [dummy, dummy+2]
            dummy += 1
            continue
        if i == rank-1:
            indices += [dummy, dummy+2], [dummy+1, dummy+2]
            break
        indices += [dummy, dummy+2, dummy+3], [dummy+1, dummy+2, dummy+4]
        dummy += 3

    return ncon(mps, indices)

def retrieve(nodes, bitstring):

    rank = len(nodes)
    indices = []
    dummy = 1
    for i in range(rank):

        if i == 0:
            indices += [[-(i+1), dummy]]
            continue
        if i == rank-1:
            indices += [[dummy, -(i+1)]]
            break
        indices += [[dummy, -(i+1), dummy+1]]
        dummy += 1

    return ncon(nodes, indices)[bitstring]

def canonicalise(d=2, D=9, rank=7, bitstring = (0,1,0,1,0)):
    """ 
    Legend: 
    first:  nodes[site][physical, virtual]
    middle: nodes[site][virtual, physical, virtual]
    last:   nodes[site][virtual, physical] 
    """

    nodes = [rnd(d,D), *[rnd(D,d,D) for i in range(rank-2)], rnd(D,d)] 

    nrm = norm(nodes)
    cmp = retrieve(nodes, bitstring)

    # L -> R
    for i in range(rank-1):
        
        if not i == 0:
            nodes[i] = nodes[i].reshape(nodes[i].shape[0]*nodes[i].shape[1],D)

        # svd
        svd = np.linalg.svd(nodes[i])
        s = np.zeros(nodes[i].shape)
        np.fill_diagonal(s, svd[1])

        if i == 0:
            # contract to form new nodes from svd
            nodes[i] = svd[0]
            nodes[i+1] = np.einsum("ij,jk,klm->ilm", s, svd[2], nodes[i+1])

        elif i == rank-2:
            # contract to form new nodes from svd
            nodes[i] = svd[0].reshape(nodes[i-1].shape[-1],d,-1) 
            nodes[i+1] = np.einsum("ij,jk,kl->il", s, svd[2], nodes[i+1])

        else:
            # contract to form new nodes from svd
            nodes[i] = svd[0].reshape(nodes[i-1].shape[-1],d,-1) 
            nodes[i+1] = np.einsum("ij,jk,klm->ilm", s, svd[2], nodes[i+1])

        # checks
        assert np.isclose(nrm, norm(nodes))
        assert np.allclose(cmp, retrieve(nodes, bitstring))

    # checks
    assert np.allclose(left_env(nodes, 3), np.eye(left_env(nodes, 3).shape[0]))
    assert np.isclose(nrm, norm(nodes))
    assert np.allclose(cmp, retrieve(nodes, bitstring))

    # R -> L
    for i in range(rank-1, 0, -1):

        if i == rank-1:
            # svd the evironment to the right of site i
            R = nodes[i] @ nodes[i].T.conj()

        else:
            # svd the evironment to the right of site i
            R = ncon([nodes[i], nodes[i].conj(), R], [[-1,2,3], [-2,2,4], [3,4]])

        svd = np.linalg.svd(R)
        s = np.zeros(R.shape)
        np.fill_diagonal(s, svd[1])

        if i == rank-1:
            nodes[i] = np.einsum("ij,jk->ik", svd[2], nodes[i])
            nodes[i-1] = np.einsum("ijk,kl->ijl", nodes[i-1], svd[0])

        elif i == 1:
            nodes[i] = np.einsum("ij,jkl->ikl", svd[2], nodes[i])
            nodes[i-1] = np.einsum("ij,jk->ik", nodes[i-1], svd[0])

        else:
            nodes[i] = np.einsum("ij,jkl->ikl", svd[2], nodes[i])
            nodes[i-1] = np.einsum("ijk,kl->ijl", nodes[i-1], svd[0])

        R = svd[2] @ R @ svd[0]

    # checks
    assert np.isclose(nrm, norm(nodes))
    assert np.allclose(cmp, retrieve(nodes, bitstring))

    return nodes
