import numpy as np
import json


def load_model(path="src/models/audio_model.json"):
    with open(path, "r") as f:
        return json.load(f)


def gaussian_prob(x, mu, sigma):
    sigma = max(sigma, 1e-6)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mu) ** 2) / (2 * sigma * 2)
    )


def state_probability(features, params):
    rms, pitch_mean, pitch_std, zcr = features

    return (
        gaussian_prob(rms, params["rms_mu"], params["rms_sigma"])
        * gaussian_prob(pitch_mean, params["pitch_mean_mu"], params["pitch_mean_sigma"])
        * gaussian_prob(pitch_std, params["pitch_std_mu"], params["pitch_std_sigma"])
        * gaussian_prob(zcr, params["zcr_mu"], params["zcr_sigma"])
    )


def classify(features, model):
    probs = {}
    for state, params in model.items():
        probs[state] = state_probability(features, params)

        # normalize
        total = sum(probs.values()) + 1e-8
        for k in probs:
            probs[k] /= total

        return probs
