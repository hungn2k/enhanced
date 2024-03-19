import os

SHADOW_REMOVE = {
    "WINDOW_SIZE": int(os.environ.get("ENHANCED_WINDOW_SIZE", 15)),
    "C1": int(os.environ.get("ENHANCED_C1", 255)),
    "C0": int(os.environ.get("ENHANCED_C0", -10)),
    "EPSILON": float(os.environ.get("ENHANCED_EPSILON", 0.1)),
    "DEFAULT_FILTER_NAME": os.environ.get("FILTER_NAME", "fbs"),
    # BC filter
    "REGULARIZE_LAMBDA": float(os.environ.get("ENHANCED_REGULARIZE_LAMBDA", 1)),
    "SIGMA": float(os.environ.get("ENHANCED_SIGMA", 0.5)),
    "DELTA": float(os.environ.get("ENHANCED_DELTA", 0.85)),
    # FBS filter
    "FILTER_LAMBDA": int(os.environ.get("ENHANCED_FILTER_LAMBDA", 250)),
    "FILTER_SIGMA_XY": int(os.environ.get("ENHANCED_FILTER_SIGMA_XY", 10)),
    "FILTER_SIGMA_L": int(os.environ.get("ENHANCED_FILTER_SIGMA_L", 50)),
    # GUIDED_FILTER
    "FILTER_R": int(os.environ.get("ENHANCED_FILTER_R", 8)),
    "FILTER_EPS": float(os.environ.get("ENHANCED_FILTER_EPS", 0.05))
}

SERVICE = {
    "SERVER_PORT": os.environ.get("PORT", "8080"),
    "SERVER_HOST": os.environ.get("HOST", "0.0.0.0")
}