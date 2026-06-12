"""Gradium transcription plugin — bundled, auto-loaded.

Mirrors the ``plugins/image_gen/openai/`` and ``plugins/browser/<vendor>/``
layout: ``provider.py`` holds the provider class; ``__init__.py::register``
instantiates and registers it via the plugin context.

First in-tree consumer of the transcription plugin hook
(``PluginContext.register_transcription_provider``).
"""

from __future__ import annotations

from plugins.transcription.gradium.provider import GradiumTranscriptionProvider


def register(ctx) -> None:
    """Register the Gradium provider with the plugin context."""
    ctx.register_transcription_provider(GradiumTranscriptionProvider())
