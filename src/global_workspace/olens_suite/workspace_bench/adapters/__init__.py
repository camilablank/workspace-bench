"""Adapters: native eval outputs -> the normalized :mod:`..schema` bundle.

One module per source format. Each exposes a ``build_*`` function returning
``(list[FamilySummary], dict[family -> list[Question]])`` so the normalizer can merge many
sources into one bundle.
"""
