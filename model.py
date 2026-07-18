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

# Step 11 - softmax_probabilities
import jax.numpy as jnp

def softmax_probabilities(logits):
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(shifted)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # Pass through all hidden layers
    for layer in params[:-1]:
        x = linear_forward(x, layer)
        x = relu_activation(x)

    # Final output layer (no activation)
    logits = linear_forward(x, params[-1])

    return logits

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # Shift logits by the maximum value for numerical stability
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)

    # Compute log(sum(exp(shifted)))
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))

    # Return log-softmax values
    return shifted - log_sum_exp

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    # Compute log probabilities using the numerically stable log-softmax
    log_probs = log_softmax_logits(logits)

    # Compute cross-entropy loss for each sample
    loss = -jnp.sum(one_hot_targets * log_probs, axis=-1)

    # Return the mean loss over the batch
    return jnp.mean(loss)

# Step 15 - classification_accuracy
import jax.numpy as jnp

def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    
    # Predict the class with the highest logit
    predictions = jnp.argmax(logits, axis=-1)

    # Compare predictions with true labels and compute the mean accuracy
    return jnp.mean(predictions == labels)

# Step 16 - loss_fn_of_params
import jax
import jax.numpy as jnp

def loss_fn_of_params(params, x, one_hot_targets):
    # Forward pass through the MLP to get logits
    logits = mlp_forward(params, x)

    # Compute and return the scalar cross-entropy loss
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
import jax
import jax.numpy as jnp

def compute_param_grads(params, x, one_hot_targets):
    # Compute gradients of the loss with respect to params
    grads = jax.grad(loss_fn_of_params)(params, x, one_hot_targets)

    return grads

# Step 18 - sgd_update_params
import jax
import jax.numpy as jnp

def sgd_update_params(params, grads, learning_rate):
    # Create a new list to store the updated parameters
    updated_params = []

    # Update each layer's weights and biases
    for param, grad in zip(params, grads):
        updated_layer = {
            "W": param["W"] - learning_rate * grad["W"],
            "b": param["b"] - learning_rate * grad["b"],
        }
        updated_params.append(updated_layer)

    return updated_params

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

