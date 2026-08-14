"""Unsupervised oracle lens: an NLA-style natural language autoencoder, no teacher/dictionary.

A workspace verbalizer (WV) maps a layer-44 activation to text; a long-span activation
reconstructor (AR) maps text back to the activation. The AR is trained FIRST on natural
assistant-text spans (1..1024 tokens) and then frozen forever; the WV is trained with GRPO
against the frozen AR (reward = negative whitened MSE) on the official NLA stack (Miles+SGLang).

Reference: https://transformer-circuits.pub/2026/nla/ and
https://github.com/kitft/natural_language_autoencoders. Deviations from the reference are
recorded in docs/project/experiments/ola/runbook.md.
"""
