# Interactive project presentation

This folder contains a ten-minute interactive presentation built with [Lectrace](https://praisegee.github.io/lectrace/). It presents the PneumoniaMNIST augmentation experiment as an executable Python lecture rather than a PowerPoint deck.

## Run locally

From the repository root, create or activate a Python 3.11+ environment, then run:

```bash
pip install -r lectures/requirements.txt
lectrace serve lectures/01_pneumonia_robustness.py
```

Open the local URL printed by Lectrace. Use the right and left arrow keys to advance or revisit steps. The inline callouts contain speaker cues and suggested timing. The lecture has nine main sections and roughly ten minutes of content; it is intentionally interactive rather than a fixed-slide PDF.

## Build for GitHub Pages

Run the following commands at the repository root:

```bash
lectrace init
lectrace build --output _site
```

`lectrace init` adds the GitHub Actions workflow and `lectrace.toml`. Commit those files, push to GitHub, then enable GitHub Pages with **GitHub Actions** as the source in repository settings.

## Contents

- `01_pneumonia_robustness.py`: the interactive ten-minute presentation
- `figures/pneumoniamnist_examples.png`: reproducible representative training images used in the data section

The lecture reports single-seed results and clearly marks this as a benchmark robustness study, not a clinical validation.
