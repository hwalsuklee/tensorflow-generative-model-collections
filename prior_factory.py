"""
Most codes from https://github.com/musyoku/adversarial-autoencoder/blob/master/sampler.py
"""

import numpy as np
from math import sin,cos,sqrt

def onehot_categorical(batch_size, n_labels):
    y = np.zeros((batch_size, n_labels), dtype=np.float32)
    indices = np.random.randint(0, n_labels, batch_size)
    for b in range(batch_size):
        y[b, indices[b]] = 1
    return y

def uniform(batch_size, n_dim, n_labels=10, minv=-1, maxv=1, label_indices=None):
    if label_indices is not None:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(label, n_labels):
            num = int(np.ceil(np.sqrt(n_labels)))
            size = (maxv-minv)*1.0/num
            x, y = np.random.uniform(-size/2, size/2, (2,))
            i = label / num
            j = label % num
            x += j*size+minv+0.5*size
            y += i*size+minv+0.5*size
            return np.array([x, y]).reshape((2,))

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
    else:
        z = np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)
    return z

def gaussian(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x-mean) + 1j*(y-mean), deg=True)

            label = ((int)(n_labels*angle))//360

            if label<0:
                label+=n_labels

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size, 1), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    a_sample, a_label = sample(n_labels)
                    z[batch, zi*2:zi*2+2] = a_sample
                    z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z

def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z