# Golden files

Frozen outputs of the source repo's own scripts (Camila's private `global-workspace`, 2026-09-15..17
state): the option lists, rendered prompts, marker offsets and regex verdicts each family's tests
compare the port against, byte for byte. They are inputs to `tests/test_<family>.py`, never
regenerated here.

The scripts that produced them (`make_<family>.py`, importing the source repo by an absolute
path) were removed in the 2026-09-23 audit because nobody outside that machine could run them;
they are in the git history before that commit if a golden file ever has to be rebuilt.
