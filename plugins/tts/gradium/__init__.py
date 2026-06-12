"""Gradium TTS plugin — bundled, auto-loaded.

Mirrors the ``plugins/image_gen/openai/`` and ``plugins/browser/<vendor>/``
layout: ``provider.py`` holds the provider class; ``__init__.py::register``
instantiates and registers it via the plugin context.

First in-tree consumer of the TTS plugin hook from issue #30398.
"""

from __future__ import annotations

from plugins.tts.gradium.provider import GradiumTTSProvider


def register(ctx) -> None:
    """Register the Gradium provider with the plugin context."""
    ctx.register_tts_provider(GradiumTTSProvider())
