import numpy as np
import json


# 🔹 Load trained model
def load_model(path="src/models/audio_model.json"):
    with open(path, "r") as f:
        return json.load(f)


# 🔹 Log Gaussian (numerically stable)
def log_gaussian_prob(x, mu, sigma):
    sigma = max(sigma, 1e-6)

    return -0.5 * np.log(2 * np.pi * sigma**2) - ((x - mu) ** 2) / (2 * sigma**2)


# 🔹 Compute log probability of a state
def state_log_probability(features, params):
    rms, pitch_mean, pitch_std, zcr = features

    return (
        log_gaussian_prob(rms, params["rms_mu"], params["rms_sigma"])
        + log_gaussian_prob(
            pitch_mean, params["pitch_mean_mu"], params["pitch_mean_sigma"]
        )
        + log_gaussian_prob(
            pitch_std, params["pitch_std_mu"], params["pitch_std_sigma"]
        )
        + log_gaussian_prob(zcr, params["zcr_mu"], params["zcr_sigma"])
    )


# 🔹 Convert log probs → normal probs (softmax)
def softmax(log_probs):
    max_log = max(log_probs.values())  # for stability

    exp_probs = {}
    total = 0

    for k, v in log_probs.items():
        exp_probs[k] = np.exp(v - max_log)
        total += exp_probs[k]

    for k in exp_probs:
        exp_probs[k] /= total

    return exp_probs


# 🔹 Main classifier
def classify(features, model):
    log_probs = {}

    for state, params in model.items():
        log_probs[state] = state_log_probability(features, params)

    # normalize using softmax
    probs = softmax(log_probs)

    return probs


# 🔹 Helper: get best state
def predict(features, model):
    probs = classify(features, model)
    return max(probs, key=probs.get), probs
