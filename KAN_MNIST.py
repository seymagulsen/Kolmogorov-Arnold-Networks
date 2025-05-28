
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from jax import grad, value_and_grad, jit
import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import gcs_utils
import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt
from jax import tree_util
import wandb

# Disable GCS for TFDS
gcs_utils._is_gcs_disabled = True

# Constants
MEAN = [0.13]
STD = [0.30]

# Set TensorFlow configuration
tf.random.set_seed(0)
tf.config.experimental.set_visible_devices([], 'GPU')

# Initialize W&B
wandb.init(
    project="KAN_vs_MLP_MNIST",
    name="batch_logging",
    config={
        "epochs": 3,
        "batch_size": 128,
        "learning_rate_kan": 3e-5,
        "learning_rate_mlp": 1e-4,
    }
)

# Load and preprocess MNIST dataset using TFDS
@tf.autograph.experimental.do_not_convert
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = (x - MEAN) / STD
    x = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x)) * 2 - 1
    x = tf.reshape(x, (-1, 28 * 28))
    y = tf.one_hot(y, depth=10)
    return x, y


@tf.autograph.experimental.do_not_convert
def prepare(ds):  
    ds = ds.shuffle(500)
    ds = ds.batch(128, drop_remainder=True)  # Batch the data
    ds = ds.map(preprocess, tf.data.experimental.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return tfds.as_numpy(ds)

# Load datasets
ds_train, info = tfds.load('mnist', split='train', shuffle_files=False, as_supervised=True, with_info=True)
ds_test, info = tfds.load('mnist', split='test', shuffle_files=False, as_supervised=True, with_info=True)

train_ds = prepare(ds_train)
test_ds = prepare(ds_test)


@partial(jax.jit, static_argnums=(2,))
def simple_bspline_basis(x_ext, grid, k=3):
    """
    Compute B-spline basis functions.

    Args:
        x_ext (jnp.ndarray): Input data.
        grid (jnp.ndarray): Grid points for B-spline.
        k (int): Degree of the B-spline.

    Returns:
        jnp.ndarray: B-spline basis functions.
    """
    def safe_divide(a, b):
        return a / (b + 1e-5)  # Handle division by zero

    grid = jnp.expand_dims(grid, axis=0)  # Shape (1, G+2k+1)
    x = jnp.expand_dims(x_ext, axis=1)    # Shape (batch_size, 1)

    # Base case for k=0
    basis_splines = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).astype(float)

    # Recursive computation for higher-order splines
    for K in range(1, k+1):
        left_term = safe_divide(x - grid[:, :-(K + 1)], grid[:, K:-1] - grid[:, :-(K + 1)])
        right_term = safe_divide(grid[:, K + 1:] - x, grid[:, K + 1:] - grid[:, 1:(-K)])
        basis_splines = left_term * basis_splines[:, :-1] + right_term * basis_splines[:, 1:]
    return basis_splines

class SimpleKANLayer(nn.Module):
    input_dim: int
    output_dim: int
    num_intervals: int
    k: int = 3

    def setup(self):
        """
        Setup the B-spline grid and initialize parameters.
        """
        # B-spline grid
        grid = jnp.linspace(-1, 1, self.num_intervals + 1)
        self.grid = jnp.concatenate([grid[:self.k], grid, grid[-self.k:]])
        # Spline coefficients
        self.coefficients = self.param('coefficients', nn.initializers.normal(0.01),
                                       (self.input_dim, self.output_dim, len(self.grid) - self.k - 1))

        # Weights for residual activation (wb and ws)
        self.wb = self.param("wb", nn.initializers.ones, (self.output_dim,))
        self.ws = self.param("ws", nn.initializers.constant(1.0), (self.output_dim,))

    def __call__(self, x):
        """
        Forward pass for the SimpleKANLayer.

        Args:
            x (jnp.ndarray): Input data.

        Returns:
            jnp.ndarray: Output after applying the layer.
        """
        x = jnp.atleast_2d(x)

        # Normalize input to [-1, 1]
        x = (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x) + 1e-6) * 2 - 1
        x = jnp.clip(x, -1, 1)
        
        # Compute spline output
        spline_outputs = []
        for i in range(self.input_dim):
            basis = simple_bspline_basis(x[:, i], self.grid, self.k)
            spline_out = jnp.dot(basis, self.coefficients[i].T)
            spline_outputs.append(spline_out)
        
        spline_result = jnp.sum(jnp.stack(spline_outputs, axis=1), axis=1)
        
        # Apply SiLU activation (element-wise)
        silu_result = x / (1 + jnp.exp(-x)) @ jnp.ones((self.input_dim, self.output_dim)) # Match output_dim

        # Combine SiLU and Spline with weights
        residual_activation = self.wb * silu_result + self.ws * spline_result
        return residual_activation

class SimpleKAN(nn.Module):
    input_dim: int
    output_dim: int
    num_intervals: int
    layer_dims: list
    k: int = 3
    dropout_rate: float = 0.2

    def setup(self):
        """
        Setup the layers for the SimpleKAN model.
        """
        self.layers = [
            SimpleKANLayer(input_dim=self.input_dim if i == 0 else self.layer_dims[i - 1],
                           output_dim=dim,
                           num_intervals=self.num_intervals,
                           k=self.k,
                           )
            for i, dim in enumerate(self.layer_dims)
        ]
        self.out_layer = nn.Dense(self.output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=False):
        """
        Forward pass for the SimpleKAN model.

        Args:
            x (jnp.ndarray): Input data.
            training (bool): Whether the model is in training mode.

        Returns:
            jnp.ndarray: Output after applying the model.
        """
        for layer in self.layers:
            x = layer(x)
            if training:
                x = self.dropout(x, deterministic=not training)
        return self.out_layer(x)

class SimpleMLP(nn.Module):
    input_dim: int
    output_dim: int
    hidden_dims: list
    dropout_rate: float = 0.2 

    def setup(self):
        """
        Setup the layers for the SimpleMLP model.
        """
        self.layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.out_layer = nn.Dense(self.output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, training=False):
        """
        Forward pass for the SimpleMLP model.

        Args:
            x (jnp.ndarray): Input data.
            training (bool): Whether the model is in training mode.

        Returns:
            jnp.ndarray: Output after applying the model.
        """
        for layer in self.layers:
            x = nn.relu(layer(x))
            if training:
                x = self.dropout(x, deterministic=not training)
        return self.out_layer(x)

# Initialize Models
kan_model = SimpleKAN(input_dim=28*28, output_dim=10, num_intervals=8, layer_dims=[32, 16])
mlp_model = SimpleMLP(input_dim=28*28, output_dim=10, hidden_dims=[32, 16])

kan_params = kan_model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
mlp_params = mlp_model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))


# Optimizer with AdamW for KAN and MLP --> the optimizer setup does not exist in the article.

scheduler = optax.exponential_decay(
    init_value=5e-4,
    transition_steps=3000,
    decay_rate=0.85,
    end_value=1e-5
)

opt_kan = optax.adamw(learning_rate=scheduler, weight_decay=0.0001)
opt_mlp = optax.adamw(learning_rate=1e-4, weight_decay=0.0001)

# Initialize Optimizer States
opt_state_kan = opt_kan.init(kan_params)
opt_state_mlp = opt_mlp.init(mlp_params)


def train_one_step(params, opt_state, x, y, key, model, optimizer, training=True):
    # Convert one-hot labels to integer class indices
    y_int = jnp.argmax(y, axis=-1)

    def loss_fn(params):
        logits = model.apply(params, x, training=True, rngs={'dropout': key})
        return optax.softmax_cross_entropy_with_integer_labels(logits, y_int).mean()

    loss_val, grads = value_and_grad(loss_fn)(params)
    # Apply gradient clipping
    grads = tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss_val

def evaluate(params, x, y, model):
    logits = model.apply(params, x, training=False)
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(predictions == jnp.argmax(y, axis=-1))
    return accuracy

# Training Loop
main_key = jax.random.PRNGKey(0)
num_batches = len(train_ds)  # Estimate the number of batches in the dataset

# Store epoch-wise metrics
epoch_kan_loss = []
epoch_mlp_loss = []
epoch_kan_acc = []
epoch_mlp_acc = []

for epoch in range(5):
    kan_loss = []
    mlp_loss = []
    main_key, sub_key = jax.random.split(main_key)
    print(f"Epoch {epoch + 1}:")

    # Training Step
    for batch_idx, batch in enumerate(train_ds):
        x_batch, y_batch = batch
        sub_key_kan, sub_key_mlp = jax.random.split(sub_key)

        # Train KAN model
        kan_params, opt_state_kan, batch_kan_loss = train_one_step(
            kan_params, opt_state_kan, jnp.array(x_batch), jnp.array(y_batch), sub_key_kan, kan_model, opt_kan
        )
        kan_loss.append(batch_kan_loss)

        # Train MLP model
        mlp_params, opt_state_mlp, batch_mlp_loss = train_one_step(
            mlp_params, opt_state_mlp, jnp.array(x_batch), jnp.array(y_batch), sub_key_mlp, mlp_model, opt_mlp
        )
        mlp_loss.append(batch_mlp_loss)

        # Print Batch Losses
        print(f"  Batch {batch_idx + 1}/{num_batches}:")
        print(f"    KAN Batch Loss: {batch_kan_loss:.4f}")
        print(f"    MLP Batch Loss: {batch_mlp_loss:.4f}")
        
        # Batch Logging
        wandb.log({
            "Batch": batch_idx + 1,
            "KAN Batch Loss": float(batch_kan_loss),
            "MLP Batch Loss": float(batch_mlp_loss),
            "Epoch": epoch + 1
        })

    # Evaluation Step
    kan_accuracies = []
    mlp_accuracies = []

    for batch in test_ds:
        x_test, y_test = batch
        kan_acc = evaluate(kan_params, jnp.array(x_test), jnp.array(y_test), kan_model)
        mlp_acc = evaluate(mlp_params, jnp.array(x_test), jnp.array(y_test), mlp_model)

        kan_accuracies.append(kan_acc)
        mlp_accuracies.append(mlp_acc)

    # Compute epoch metrics
    kan_mean_loss = jnp.mean(jnp.array(kan_loss))
    mlp_mean_loss = jnp.mean(jnp.array(mlp_loss))
    kan_mean_acc = jnp.mean(jnp.array(kan_accuracies))
    mlp_mean_acc = jnp.mean(jnp.array(mlp_accuracies))

    epoch_kan_loss.append(kan_mean_loss)
    epoch_mlp_loss.append(mlp_mean_loss)
    epoch_kan_acc.append(kan_mean_acc)
    epoch_mlp_acc.append(mlp_mean_acc)

    # Print Epoch Summary
    print(f"Epoch {epoch + 1} Summary:")
    print(f"  KAN Loss: {kan_mean_loss:.4f}")
    print(f"  MLP Loss: {mlp_mean_loss:.4f}")
    print(f"  KAN Accuracy: {kan_mean_acc:.4%}")
    print(f"  MLP Accuracy: {mlp_mean_acc:.4%}")

    # Epoch Logging
    wandb.log({
        "Epoch": epoch + 1,
        "KAN Loss": kan_mean_loss,
        "MLP Loss": mlp_mean_loss,
        "KAN Accuracy": kan_mean_acc,
        "MLP Accuracy": mlp_mean_acc
    })

wandb.finish()

# Plot Loss
plt.figure()
plt.plot(epoch_kan_loss, label='KAN Loss')
plt.plot(epoch_mlp_loss, label='MLP Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epoch_kan_acc, label='KAN Accuracy')
plt.plot(epoch_mlp_acc, label='MLP Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()
plt.show()