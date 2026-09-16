# TODO

Open work only; merged work is in the git log and the README.

- [ ] Judge the arms across every family (s3d RL600, s3d SFT251, NLA, NLA SFT, J-lens, R-Lens,
      logit lens, template lens) with `wsbench run`, read against the frozen floors.
- [ ] Revisit the multihop_mt and basic_readout_mt option pools: blind lucky guessing picks the
      gold 0.66 and 0.46 of the time from option shape alone (uniform 0.13 / 0.14).
- [ ] Decide whether the single-token basics with high prompt-only floors (typo 1.00, multihop
      0.90, multilingual 0.76, basic_readout 0.72, poetry 0.71) stay headline evals.
- [ ] Move `evals/jlens_concept_pr/gen-jlens-pr-jlens/` (4.6 MB, 3.6k files of producer output)
      out of git.
- [ ] Per-family empirical nulls where a family has none.
