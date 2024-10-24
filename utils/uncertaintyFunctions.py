import equinox as eqx
import jax
import jax.numpy as jnp


@eqx.filter_jit
def computeUncertainty(predictions, ood_predictions):
    # Predictions are given as direct outputs of the model (no softmax)
    # shape is (batch_size, )
    # Labels are one-hot encoded
    softmax_predictions = jax.nn.softmax(predictions, axis=-1)
    softmax_ood_predictions = jax.nn.softmax(ood_predictions, axis=-1)
    def compute_full_uncertainty(predictions):
        """
        Compute the aleatoric and epistemic uncertainty for all classes.

        Args:
            predictions (jax.numpy.ndarray): Predictions of the model (shape (n_elements, n_samples, n_classes))

        Returns:
            aleatoric_uncertainty (jax.numpy.ndarray): Aleatoric uncertainty (shape (n_elements, n_classes))
            epistemic_uncertainty (jax.numpy.ndarray): Epistemic uncertainty (shape (n_elements, n_classes))
        """
        # Compute the mean predictions over the sample dimension (n_samples)
        mean_predictions = jnp.mean(predictions, axis=1)
        # Compute total uncertainty (entropy of mean predictions)
        total_uncertainty = -mean_predictions * \
            jnp.log2(mean_predictions + 1e-8)
        # Sum over classes for total uncertainty per element
        total_uncertainty = jnp.sum(total_uncertainty, axis=-1)
        # Compute epistemic uncertainty using KL divergence across all samples

        def compute_kl_per_element(predictions_per_element):
            # predictions_per_element has shape (n_samples, n_classes) for each element
            mean_pred = jnp.mean(predictions_per_element,
                                 axis=0)  # Shape: (n_classes)

            def compute_kl_per_sample(pred):
                # KL divergence between predictions and mean predictions
                kl_div = pred * (jnp.log2(pred + 1e-8) -
                                 jnp.log2(mean_pred + 1e-8))
                return jnp.sum(kl_div, axis=-1)  # Sum over classes
            # Apply KL computation to all samples for the current element
            return jnp.mean(jax.vmap(compute_kl_per_sample)(predictions_per_element), axis=0)
        # Use vmap to compute epistemic uncertainty over all elements
        epistemic_uncertainty = jax.vmap(compute_kl_per_element)(
            predictions)  # Shape: (n_elements)
        # Aleatoric uncertainty is the difference between total and epistemic uncertainty
        aleatoric_uncertainty = total_uncertainty - \
            epistemic_uncertainty
        return aleatoric_uncertainty, epistemic_uncertainty
    
    # Compute the aleatoric and epistemic uncertainty for each class
    aleatoric_uncertainty, epistemic_uncertainty = jax.vmap(compute_full_uncertainty)(
        softmax_predictions)
    aleatoric_uncertainty_ood, epistemic_uncertainty_ood = jax.vmap(compute_full_uncertainty)(
        softmax_ood_predictions)

    def compute_roc_auc(epistemic_uncertainty, epistemic_uncertainty_ood):

        # Compute the true positive rate and false positive rate
        x = jnp.linspace(0, 1, 100)
        tpr = jnp.array([jnp.mean(epistemic_uncertainty_ood > threshold)
                         for threshold in x])
        fpr = jnp.array([jnp.mean(epistemic_uncertainty > threshold)
                         for threshold in x])
        # Compute AUC using the trapezoidal rule
        auc = -jnp.trapezoid(y=tpr, x=fpr)
        return auc

    auc = compute_roc_auc(epistemic_uncertainty, epistemic_uncertainty_ood)
    return aleatoric_uncertainty, epistemic_uncertainty, aleatoric_uncertainty_ood, epistemic_uncertainty_ood, auc
