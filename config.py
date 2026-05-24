# config.py — Configuration Dreamer World Family

OUTPUT_DIR = "outputs"
OUTPUT_DREAMER_XL   = "outputs/dreamer_xl_scene.png"
OUTPUT_XDREAMER     = "outputs/xdreamer_scene.png"
OUTPUT_ATD          = "outputs/atd_navmap.png"
OUTPUT_COMPARISON   = "outputs/comparison.png"
OUTPUT_REPORT       = "outputs/scene_report.json"

RENDER_DPI      = 150
RENDER_FIGSIZE  = (10, 7)

# Palettes par environnement
ENVIRONMENT_PALETTES = {
    "futuristic city":       {"bg": "#0a0f2c", "accent": "#00d4ff", "ground": "#1a1a2e"},
    "mountain landscape":    {"bg": "#c8e6f5", "accent": "#4a6fa5", "ground": "#6b8e5a"},
    "forest":                {"bg": "#1a3a1a", "accent": "#52c41a", "ground": "#2d5a1b"},
    "outer space":           {"bg": "#000010", "accent": "#ffffff", "ground": "#1a0030"},
    "interior room":         {"bg": "#f5e6d0", "accent": "#8b6914", "ground": "#d4a96a"},
    "desert landscape":      {"bg": "#f4d03f", "accent": "#e67e22", "ground": "#d4a96a"},
    "underwater scene":      {"bg": "#0d4f8b", "accent": "#00ffcc", "ground": "#0a3a6b"},
    "simple digital environment": {"bg": "#1a1a2e", "accent": "#6c63ff", "ground": "#16213e"},
}
