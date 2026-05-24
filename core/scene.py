# core/scene.py — Scene data model for Dreamer World Family Demo

from dataclasses import dataclass, asdict, field
from typing import List
import json


@dataclass
class Scene:
    """
    Structured internal representation of an imagined world.

    Produced by the WorldModelAgent and consumed by the RenderAgent.
    Mirrors the latent world state concept used in real Dreamer systems.

    Attributes:
        prompt:       The enriched input prompt.
        main_object:  Primary subject of the scene.
        environment:  Spatial context and background setting.
        style:        Visual aesthetic style.
        colors:       Dominant color palette detected from the prompt.
        camera:       Camera angle / point of view.
        objects:      Full list of scene elements (main + secondary).
        confidence:   Agent confidence in scene extraction (0.0–1.0).
    """

    prompt: str
    main_object: str
    environment: str
    style: str
    colors: List[str]
    camera: str
    objects: List[str]
    confidence: float = field(default=1.0)

    def to_pretty_text(self) -> str:
        """Return a formatted multi-line string representation."""
        colors_str = ", ".join(self.colors) if self.colors else "N/A"
        objects_str = ", ".join(self.objects) if self.objects else "N/A"

        return (
            f"\n"
            f"    Prompt       : {self.prompt}\n\n"
            f"    Imagined World:\n"
            f"      Main object  : {self.main_object}\n"
            f"      Environment  : {self.environment}\n"
            f"      Style        : {self.style}\n"
            f"      Camera       : {self.camera}\n"
            f"      Colors       : {colors_str}\n"
            f"      Objects      : {objects_str}\n"
            f"      Confidence   : {self.confidence:.0%}\n"
        )

    def to_dict(self) -> dict:
        """Return scene as a plain dictionary (for JSON export)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize scene to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
