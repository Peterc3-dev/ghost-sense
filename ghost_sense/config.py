"""Configuration: decay rates, signal weights, thresholds."""

# Decay lambdas per signal type (used in confidence * e^(-lambda * delta_hours))
# Higher lambda = faster decay = less persistent signal
DECAY_LAMBDAS: dict[str, float] = {
    # Register signals decay fast — they reflect the current message style
    "register.slang_ratio": 0.5,
    "register.avg_word_length": 0.5,
    "register.punctuation_density": 0.5,
    "register.sentence_count": 0.5,
    "register.capitalization_ratio": 0.5,
    "register.emoji_density": 0.5,
    "register.formality_score": 0.3,
    # Sleep signals persist longer
    "sleep.quality": 0.05,
    "sleep.caffeine_proxy": 0.1,
    "sleep.energy_level": 0.15,
    # Nutrition
    "nutrition.last_meal": 0.08,
    "nutrition.quality": 0.06,
    # Stress
    "stress.level": 0.1,
    "stress.source": 0.07,
    # Cadence
    "cadence.burst_score": 0.4,
    "cadence.avg_interval": 0.3,
}

# Default lambda for dimensions not explicitly listed
DEFAULT_DECAY_LAMBDA = 0.2

# Minimum confidence to consider a field "known"
CONFIDENCE_THRESHOLD = 0.15

# Absence detection calibration period (hours)
ABSENCE_CALIBRATION_HOURS = 336  # 14 days

def get_lambda(dimension: str) -> float:
    return DECAY_LAMBDAS.get(dimension, DEFAULT_DECAY_LAMBDA)
