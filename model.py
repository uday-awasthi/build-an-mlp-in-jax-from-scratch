"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import numpy as np
import jax
import jax.numpy as jnp


def make_prng_key(seed):
    np.random.seed(seed)          # Optional: for NumPy randomness
    return jax.random.PRNGKey(seed)


def split_prng_key(key, num=2):
    return jax.random.split(key, num)


seed = 0
batch_size = 32

root_key = make_prng_key(seed)
data_key, init_key = split_prng_key(root_key, 2)

print("Root Key:", root_key)
print("Data Key:", data_key)
print("Init Key:", init_key)

# Step 2 - split_prng_key
import jax

def split_prng_key(key, num):
    return jax.random.split(key, num)

def make_prng_key(seed):
    return jax.random.PRNGKey(seed)

key = make_prng_key(0)

subkeys = split_prng_key(key, 2)

print(subkeys.shape)
print(subkeys.dtype)

# Step 3 - sample_normal_matrix
import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    # TODO: draw a batch of random input feature vectors from the PRNG key
    return jax.random.normal(key, (batch_size, num_features))

# Step 5 - assign_class_labels
import jax.numpy as jnp

def assign_class_labels(inputs, num_classes):
    labels = jnp.argmax(inputs[:, :num_classes], axis=1)
    return labels.astype(jnp.int32)

# Step 6 - one_hot_encode_labels
import jax
import jax.numpy as jnp

def one_hot_encode_labels(labels, num_classes):
    return jax.nn.one_hot(labels, num_classes)

# Step 7 - init_linear_layer
import jax
import jax.numpy as jnp

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    W = jax.random.normal(key, (in_dim, out_dim)) * scale
    b = jnp.zeros((out_dim,))
    return {
        "W": W,
        "b": b
    }

# Step 8 - init_mlp_params
import jax

def init_mlp_params(key, layer_sizes, scale=0.1):
    num_layers = len(layer_sizes) - 1
    keys = jax.random.split(key, num_layers)

    params = []

    for i in range(num_layers):
        params.append(
            init_linear_layer(
                keys[i],
                layer_sizes[i],
                layer_sizes[i + 1],
                scale
            )
        )

    return params

# Step 9 - linear_forward
import jax.numpy as jnp

def linear_forward(x, layer_params):
    W = layer_params["W"]
    b = layer_params["b"]
    return jnp.matmul(x, W) + b

# Step 10 - relu_activation
import jax.numpy as jnp

def relu_activation(x):
    return jnp.maximum(x, 0)

# Step 11 - softmax_probabilities (not yet solved)
# TODO: implement

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

