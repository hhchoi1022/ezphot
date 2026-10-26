"""
ezphot.recipes - parameterized, CLI-callable photometry workflows built on ezphot objects
(ScienceImage / ImageSet / Catalog / CatalogSet). Agent-agnostic: usable by hand, by
Claude Code skills/agents, or by bridge.

    python -m ezphot.recipes.process_images   --objname EG21 --filter m400 --ra ... --dec ...
    python -m ezphot.recipes.stack <process_images summary.csv> [--ra ... --dec ...]
    python -m ezphot.recipes.dia <process_images or stack summary.csv> [--ra ... --dec ...]
    python -m ezphot.recipes.target_photometry --objname EG21 --ra ... --dec ...
    python -m ezphot.recipes.qa <summary.csv>
    python -m ezphot.recipes.visualize <summary.csv> [--ra ... --dec ...]
    python -m ezphot.recipes.status [--watch 60]
"""
from .process_images import process_image, process_images
from .stack import stack, stack_group
from .dia import dia, dia_image
from .target_photometry import target_photometry
from .qa import qa
from .visualize import visualize
from .status import write_status
