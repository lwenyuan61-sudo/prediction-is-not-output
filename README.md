# Prediction Is Not Output

**One output head cannot both predict what other people say and say what the model itself should say. Two heads can.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24054274.svg)](https://doi.org/10.5281/zenodo.24054274)

Code, run logs, tables and figures for the paper *Prediction Is Not Output: One Head Cannot Do Both, Two Heads Can* (Wenyuan Li, 2026).

[Project page](https://lwenyuan61-sudo.github.io/prediction-is-not-output/) · [PDF](paper/Prediction_Is_Not_Output.pdf) · [Zenodo](https://doi.org/10.5281/zenodo.24054274) · [aiXiv](https://aixiv.science/abs/aixiv.261108.000002)

## The finding

A language model uses one output head for two jobs: predicting what other people write, and producing its own replies. Where the people in the text are wrong, the two jobs want different outputs: a good prediction reproduces their mistakes, and a good reply does not.

Trained on both, one head can only reach a weighted mixture of the two targets, which is wrong for both. This holds for supervised fine-tuning, reinforcement learning and KL-regularized RLHF (Propositions 1 and 2).

We test this in a synthetic world where every target is known and other speakers answer correctly only half of the time. Mean over 5 seeds:

| Variant | Held-out accuracy (%) | Prediction loss after "I:" (bits; minimum 2.85) | Result |
|---|---|---|---|
| One-Seq: one head, predict then act | 97.0 | 16.45 | Forgets how to predict |
| One-Joint: one head, trained jointly | 98.8 | 8.48 | Expects others to be right 99% of the time |
| Two-Joint: two heads, trained jointly | 94.6 | 3.16 | Both jobs right |
| Two-Shared: two heads, shared unembedding | 98.1 | 3.25 | Both jobs right |

The price of separation: facts that appear only in plain statements reach a separate action head more slowly.

## Reproduce

Python 3.10 or newer. `pip install -r requirements.txt`

**Rebuild every table and figure from the logs of all runs (seconds, no GPU):**

```
python make_results.py
```

The output goes to `out/`. It matches the tables in the paper exactly.

**One run** (about 10 to 15 minutes on a consumer GPU, about an hour on one CPU core):

```
python tf_split3.py arm=D seed=1 S=4000 S3=1500 dev=cuda out=my_run.jsonl
```

Main arguments: `arm` (A to H), `seed`, `S` (steps of the main training), `S3` (steps with new people, 0 = none), `mef` (share of other speakers tagged "I:"), `ph3c=1` (new people also appear in dialogue), `p2` (second stage of B and the cross-entropy-only check: `both`, `sft`, `rl`), `kl` (KL weight for B), `aw` (action-loss weight), `d` (width), `dev` (`cpu` or `cuda`), `th` (CPU threads).

**All runs.** Finished runs are skipped, so move the shipped `runs/` folder elsewhere first:

```
python run_all.py cuda 6            # main comparison, B variants, widths, shares m
python run_all.py cuda 6 H          # variant H, 5 seeds
python run_all.py cuda 6 conf       # new people also in dialogue
python run_all.py cuda 6 mix        # shares m for C, D, H
python run_all.py cuda 6 mixbase    # baseline rows of the reversed-weight table
python run_all.py cuda 6 mixsft     # cross-entropy-only check (Appendix D)
python run_all.py cuda 6 rev        # reversed weights, aw = 1e-4
python make_results.py
```

`per_seed.csv` holds the end-of-training numbers of every run and seed; its `group` column says which table each run belongs to. Token names in the synthetic world are Chinese words; each token is only an ID, and the paper shows English translations.

## Help wanted: test it on a real model

This work uses a 1.5-million-parameter transformer in a synthetic world. **We did not train or fine-tune a pretrained language model.** That is the open question, and we would like to see the answer.

The test: take a pretrained model and give it a second output head (a copy of the final layer norm and unembedding). Keep training one head on next-token prediction of ordinary text, and train the other on the post-training objective (SFT, RL or RLHF). Compare it with the usual single head on:

- prediction of text in which people answer wrongly or disagree;
- the quality of the model's own replies;
- whether it can sample a population's distribution of answers, which instruction-tuned models fail at ([Jang et al., 2026](https://arxiv.org/abs/2607.25292)).

If you run this, please open an issue with your result, whichever way it goes.

## Variant names

The code and the run files use letters; the paper uses names.

| Code | Paper | Code | Paper |
|---|---|---|---|
| A | Two-Mid | E | Two-Frozen |
| B | One-Seq | F | Two-Stop |
| C | One-Joint | G | Two-Extra |
| D | Two-Joint | H | Two-Shared |

## How this was made

The author conceived the study, set its question and its claims, and directed the experiments. An AI assistant (Claude, Anthropic) wrote the experiment code, ran the experiments, made the figures and wrote the text under the author's direction.

## Citation

```bibtex
@misc{li2026prediction,
  title     = {Prediction Is Not Output: One Head Cannot Do Both, Two Heads Can},
  author    = {Li, Wenyuan},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24054274},
  url       = {https://doi.org/10.5281/zenodo.24054274}
}
```

## License

Code: [Apache License 2.0](LICENSE). Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
