"""Post-hoc calibrators from notebook cells 79 and 85 (zero-based).

The class order is Negative, Positive, Uncertain. Fit only on calibration
images. Numeric epsilons retain the two original notebook implementations.
Learned parameters are JSON serializable; inference does not load pickle files.
"""
from dataclasses import dataclass, field
import warnings
import numpy as np

METHODS = ("uncalibrated", "temperature_scaling", "vector_scaling", "dirichlet",
           "beta_negative", "isotonic_negative")


def validate_logits(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 3 or len(z) == 0 or not np.isfinite(z).all():
        raise ValueError("Expected a nonempty, finite (n, 3) logits array.")
    return z


def softmax(logits):
    z = validate_logits(logits)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


stable_softmax = softmax


def log_softmax(logits):
    z = validate_logits(logits)
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def normalize_probabilities(probabilities, epsilon=1e-10):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Probabilities must have three finite columns.")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities are outside [0, 1].")
    p = np.clip(p, epsilon, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _validate_labels(labels, n):
    raw = np.asarray(labels)
    if raw.shape != (n,) or not np.isin(raw, [0, 1, 2]).all():
        raise ValueError("Labels must be 0, 1, or 2 and match the logits.")
    y = raw.astype(np.int64)
    if set(y) != {0, 1, 2}:
        raise ValueError("Calibration requires all three classes.")
    return y


def multiclass_nll(probabilities, labels, epsilon=1e-10):
    p = normalize_probabilities(probabilities, epsilon)
    return float(-np.log(p[np.arange(len(p)), np.asarray(labels, dtype=int)]).mean())


def fit_temperature_scaling(logits, labels):
    from scipy.optimize import minimize_scalar
    z = validate_logits(logits)
    y = _validate_labels(labels, len(z))
    result = minimize_scalar(
        lambda log_t: multiclass_nll(softmax(z / np.exp(log_t)), y, 1e-12),
        bounds=(-5.0, 5.0), method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"Temperature scaling did not converge: {result.message}")
    return {"temperature": float(np.exp(result.x)), "log_temperature": float(result.x),
            "objective_nll": float(result.fun), "optimization_success": True,
            "optimization_message": str(result.message)}


def vector_scaling_transform(logits, scales, biases):
    return softmax(validate_logits(logits) * np.asarray(scales).reshape(1, 3)
                   + np.asarray(biases).reshape(1, 3))


def fit_vector_scaling(logits, labels, l2_strength=1e-3):
    """Unbounded original cell 85 optimization, centered biases, positive scales."""
    from scipy.optimize import minimize
    z = validate_logits(logits)
    y = _validate_labels(labels, len(z))

    def objective(theta):
        log_scales, raw_biases = theta[:3], theta[3:]
        biases = raw_biases - raw_biases.mean()
        p = vector_scaling_transform(z, np.exp(log_scales), biases)
        return multiclass_nll(p, y, 1e-10) + l2_strength * (
            np.sum(log_scales ** 2) + np.sum(biases ** 2))

    result = minimize(objective, np.zeros(6), method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Vector scaling did not converge: {result.message}")
    scales = np.exp(result.x[:3])
    biases = result.x[3:] - result.x[3:].mean()
    return {"scales": scales.tolist(), "biases": biases.tolist(),
            "l2_strength": float(l2_strength), "optimization_success": True,
            "optimization_message": str(result.message), "objective_value": float(result.fun),
            "calibration_nll_after_fit": multiclass_nll(
                vector_scaling_transform(z, scales, biases), y, 1e-10)}


def beta_features(negative_probabilities):
    p = np.clip(np.asarray(negative_probabilities), 1e-10, 1 - 1e-10)
    return np.column_stack((np.log(p), -np.log(1 - p)))


def reconstruct_multiclass_from_negative_probability(calibrated_negative, original):
    p = normalize_probabilities(original)
    q = np.clip(np.asarray(calibrated_negative), 1e-10, 1 - 1e-10)
    remaining = p[:, 1] + p[:, 2]
    positive_ratio = np.divide(p[:, 1], remaining, out=np.full(len(p), 0.5),
                               where=remaining > 1e-10)
    return normalize_probabilities(np.column_stack(
        (q, (1 - q) * positive_ratio, (1 - q) * (1 - positive_ratio))))


@dataclass
class Calibrator:
    method: str
    parameters: dict = field(default_factory=dict)

    def transform(self, logits):
        z = validate_logits(logits)
        par = self.parameters
        if self.method == "uncalibrated":
            return softmax(z)
        if self.method == "temperature_scaling":
            return softmax(z / par["temperature"])
        if self.method == "vector_scaling":
            return normalize_probabilities(vector_scaling_transform(z, par["scales"], par["biases"]))
        if self.method == "dirichlet":
            scores = log_softmax(z) @ np.asarray(par["coefficients"]).T + par["intercepts"]
            return normalize_probabilities(softmax(scores), 1e-12)
        p = normalize_probabilities(softmax(z))
        if self.method == "beta_negative":
            score = beta_features(p[:, 0]) @ np.asarray(par["coefficients"]) + par["intercept"]
            q = np.exp(-np.logaddexp(0.0, -score))
        elif self.method == "isotonic_negative":
            q = np.interp(p[:, 0], par["x_thresholds"], par["y_thresholds"])
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        return normalize_probabilities(
            reconstruct_multiclass_from_negative_probability(q, p))

    def to_dict(self):
        return {"method": self.method, "parameters": self.parameters,
                "class_order": ["Negative", "Positive", "Uncertain"],
                "fit_split": "calibration"}

    @classmethod
    def from_dict(cls, data):
        if data.get("class_order") != ["Negative", "Positive", "Uncertain"]:
            raise ValueError("Unexpected class order in saved calibrator.")
        return cls(data["method"], data["parameters"])


def fit_calibrator(method, logits, labels):
    z = validate_logits(logits)
    y = _validate_labels(labels, len(z))
    if method == "uncalibrated":
        return Calibrator(method)
    if method == "temperature_scaling":
        return Calibrator(method, fit_temperature_scaling(z, y))
    if method == "vector_scaling":
        return Calibrator(method, fit_vector_scaling(z, y))
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose from {METHODS}.")
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    p = normalize_probabilities(softmax(z))
    if method == "isotonic_negative":
        fitted = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                    out_of_bounds="clip").fit(p[:, 0], (y == 0).astype(float))
        return Calibrator(method, {"x_thresholds": fitted.X_thresholds_.tolist(),
                                   "y_thresholds": fitted.y_thresholds_.tolist()})
    fitted = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000,
                                fit_intercept=True, random_state=42 if method == "dirichlet" else 0)
    with warnings.catch_warnings():
        # Do not repeat the notebook's suppression of optimizer warnings.
        warnings.simplefilter("error", ConvergenceWarning)
        fitted.fit(log_softmax(z) if method == "dirichlet" else beta_features(p[:, 0]),
                   y if method == "dirichlet" else (y == 0).astype(int))
    if method == "dirichlet":
        if fitted.classes_.tolist() != [0, 1, 2]:
            raise ValueError("Dirichlet calibrator classes are not [0, 1, 2].")
        pars = {"coefficients": fitted.coef_.tolist(), "intercepts": fitted.intercept_.tolist()}
    else:
        pars = {"coefficients": fitted.coef_[0].tolist(), "intercept": float(fitted.intercept_[0])}
    pars.update({"C": 1.0, "max_iter": 5000, "solver": "lbfgs",
                 "iterations": fitted.n_iter_.tolist()})
    return Calibrator(method, pars)
