# Kolmogorov-Arnold Networks (KAN) vs Multi-Layer Perceptrons (MLP)

This project implements and compares **Kolmogorov-Arnold Networks (KANs)** and traditional **Multi-Layer Perceptrons (MLPs)** on the MNIST dataset using JAX, Flax, and TensorFlow. The primary goal is to explore the representational power, interpretability, and performance of KANs, which are inspired by the **Kolmogorov-Arnold Representation Theorem**.

---

## What is KAN?

Kolmogorov-Arnold Networks differ from MLPs in a fundamental way:

- **MLPs** use **fixed activation functions** on **nodes (neurons)**.
- **KANs** use **learnable activation functions (splines)** on **edges (weights)**.

This leads to:
- Higher interpretability
- Adaptive feature learning
- Improved convergence in some cases

---

##  Project Structure

- `SimpleKAN` - Implementation of KAN using B-spline basis functions.
- `SimpleMLP` - Baseline MLP for comparison.
- `train_one_step()` - Shared training function for both models.
- `evaluate()` - Evaluation function to compute test accuracy.
- `MNIST` - Preprocessed using TensorFlow Datasets (TFDS).
- `W&B` - Integrated logging of batch-wise and epoch-wise metrics.

---

##  Experimental Setup

| Feature           | Value                        |
|------------------|------------------------------|
| Dataset          | MNIST                        |
| Optimizer        | AdamW                        |
| Frameworks       | JAX, Flax, TensorFlow        |
| Epochs           | 5                            |
| Batch Size       | 128                          |
| Learning Rate    | KAN: scheduled (starts at 3e-5) <br> MLP: fixed (1e-4) |
| Dropout Rate     | 0.2                          |
| Evaluation Metric| Loss, Accuracy               |

---

##  Results Summary

- **KAN**
  - Faster convergence
  - Higher peak accuracy (~89%)
  - Slight instability in loss (spikes)
  - Sensitive to learning rate schedule

- **MLP**
  - Slower, steadier learning
  - Final accuracy ~85%
  - More stable loss curve

Graphs below show accuracy and loss trends per epoch.

---
##  Future Work

- Extend KANs to more complex datasets (e.g., CIFAR-10, ImageNet).
- Improve training efficiency and reduce computational cost.
- Enhance the interpretability of learned spline functions.
- Fine-tune hyperparameters to prevent overfitting or loss spikes.

---

##  References

- Liu, Z., et al. (2024). **KAN: Kolmogorov–Arnold Networks**.
- Kolmogorov, A. N., & Arnold, V. I. **Representation Theorem for Multivariate Continuous Functions**.

---

##  Author

**Şeyma Gülşen Akkuş**  
Graduate Student, Applied Data Science  
TED University  
GitHub: [@seymagulsen](https://github.com/seymagulsen)
