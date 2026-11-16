"""
Shared by the recipes. NothingToDo is exported as ezphot.recipes.NothingToDo (RECIPES_CONTRACT.md 1).
"""


class NothingToDo(ValueError):
    """The input leaves nothing to process: no images, no successful summary rows, no group of more than one image,
    no catalogs. Raised by the recipe functions before any processing (no summary CSV), with the messages of
    RECIPES_CONTRACT.md. A ValueError, so handlers of ValueError still catch it."""
