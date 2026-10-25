"""Interactive 10-minute project presentation for Lectrace.

Run from the repository root:
    lectrace serve lectures/01_pneumonia_robustness.py
"""

from pathlib import Path

from lectrace import image, link, note, plot, table, text


LECTURE_DIR = Path(__file__).resolve().parent


CONDITIONS = [
    "Clean",
    "Blur 1.0",
    "Blur 2.0",
    "Noise 0.03",
    "Noise 0.07",
    "Contrast 0.60",
    "Contrast 0.35",
    "Low resolution",
]

CNN_CONTROL = [0.955, 0.951, 0.927, 0.934, 0.878, 0.932, 0.781, 0.950]
CNN_AUGMENTED = [0.962, 0.959, 0.949, 0.954, 0.934, 0.952, 0.932, 0.958]
VIT_CONTROL = [0.953, 0.955, 0.956, 0.954, 0.940, 0.941, 0.834, 0.956]
VIT_AUGMENTED = [0.951, 0.953, 0.954, 0.954, 0.948, 0.923, 0.851, 0.953]

DATA_SPLITS = [
    {"Split": "Train", "Normal": 1214, "Pneumonia": 3494, "Total": 4708},
    {"Split": "Validation", "Normal": 135, "Pneumonia": 389, "Total": 524},
    {"Split": "Test", "Normal": 234, "Pneumonia": 390, "Total": 624},
]

AUGMENTATIONS = [
    {"Augmentation": "Gaussian blur", "Range": "$\\sigma=0.35$ to $1.25$ pixels"},
    {"Augmentation": "Gaussian noise", "Range": "SD 0.01 to 0.05"},
    {"Augmentation": "Contrast reduction", "Range": "factor 0.60 to 1.00"},
    {"Augmentation": "Reduced resolution", "Range": "50% to 85%, then upsample"},
]

CLEAN_RESULTS = [
    {"Metric": "AUROC", "C0": 0.955, "C1": 0.962, "T0": 0.953, "T1": 0.951},
    {"Metric": "AUPRC", "C0": 0.967, "C1": 0.974, "T0": 0.969, "T1": 0.968},
    {"Metric": "Balanced accuracy", "C0": 0.813, "C1": 0.821, "T0": 0.867, "T1": 0.839},
    {"Metric": "Macro-F1", "C0": 0.833, "C1": 0.842, "T0": 0.877, "T1": 0.856},
]

MEAN_DEGRADED = [
    {"Architecture": "CNN", "Control": 0.907, "Augmented": 0.948, "Change": "+0.041"},
    {"Architecture": "ViT", "Control": 0.934, "Augmented": 0.934, "Change": "-0.0001"},
]

ROBUSTNESS_ROWS = [
    {"architecture": architecture, "training": training, "condition": condition, "auroc": auroc}
    for architecture, training, scores in [
        ("CNN", "Clean-only control", CNN_CONTROL),
        ("CNN", "Quality-augmented", CNN_AUGMENTED),
        ("Vision Transformer", "Clean-only control", VIT_CONTROL),
        ("Vision Transformer", "Quality-augmented", VIT_AUGMENTED),
    ]
    for condition, auroc in zip(CONDITIONS, scores)
]

ROBUSTNESS_CHART = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"values": ROBUSTNESS_ROWS},
    "facet": {"column": {"field": "architecture", "type": "nominal", "header": {"title": None}}},
    "spec": {
        "width": 390,
        "height": 260,
        "mark": {"type": "line", "point": True, "strokeWidth": 3},
        "encoding": {
            "x": {"field": "condition", "type": "ordinal", "sort": CONDITIONS, "axis": {"labelAngle": -35, "title": "Test condition"}},
            "y": {"field": "auroc", "type": "quantitative", "scale": {"domain": [0.75, 0.98]}, "axis": {"title": "AUROC"}},
            "color": {"field": "training", "type": "nominal", "title": None, "legend": {"orient": "bottom", "direction": "horizontal"}},
            "tooltip": [
                {"field": "architecture", "type": "nominal", "title": "Model"},
                {"field": "training", "type": "nominal", "title": "Training"},
                {"field": "condition", "type": "nominal", "title": "Condition"},
                {"field": "auroc", "type": "quantitative", "format": ".3f", "title": "AUROC"},
            ],
        },
    },
}


def main():
    title()
    motivation()
    research_question()
    data()
    methods()
    evaluation()
    clean_results()
    robustness_results()
    conclusion()


def title():
    text("""
    # Can quality-aware data augmentation improve pneumonia classification robustness?

    **Yuguang Duan**  
    MD6329: Deep Learning for Healthcare AI  
    Nanyang Technological University
    """)
    note("0:00--0:30. Introduce the practical question: chest X-rays vary in quality, and a model should not fail simply because an image is blurred or low contrast.")


def motivation():
    text("""
    ## Why image quality matters

    Chest X-ray models can perform well on clean benchmark images, yet real acquisition conditions can introduce blur, noise, reduced contrast, or lower effective resolution. We therefore evaluate performance under controlled quality shifts, not clean performance alone.
    """)
    link(
        title="Generalization of deep learning models for pneumonia detection",
        authors=["Zech", "Badgeley", "Liu"],
        date="2018",
        url="https://doi.org/10.1371/journal.pmed.1002683",
        description="External validation showed that pneumonia models may not generalize consistently across hospital systems.",
    )
    note("0:30--1:30. Quality shift is only one kind of distribution shift. This project isolates it so that the comparison is reproducible.")


def research_question():
    text("""
    ## Research question and hypotheses

    **Can quality-aware augmentation improve robustness to degraded chest X-rays without sacrificing clean-image performance?**

    - **H1:** Augmentation increases the mean AUROC across seven degraded test conditions.
    - **H2:** Augmentation does not reduce clean-test AUROC by more than 0.02.
    """)
    text("""
    ### Four controlled experiments

    - C0: CNN trained on clean images
    - C1: CNN trained with quality augmentation
    - T0: ViT trained on clean images
    - T1: ViT trained with quality augmentation
    """)
    note("1:30--2:20. Emphasize the comparisons C1 versus C0 and T1 versus T0. CNN versus ViT is an architecture comparison, not the main treatment-control contrast.")


def data():
    text("""
    ## Dataset: PneumoniaMNIST

    We used 5,856 pediatric chest X-rays at $128\\times128$ resolution. Label 0 denotes normal and label 1 denotes pneumonia.
    """)
    table(DATA_SPLITS, caption="Predefined PneumoniaMNIST splits")
    image(str(LECTURE_DIR / "figures" / "pneumoniamnist_examples.png"), width="55%")
    link(
        title="MedMNIST v2",
        authors=["Yang", "et al."],
        date="2023",
        url="https://doi.org/10.1038/s41597-022-01721-8",
        description="A standardized collection of biomedical image classification datasets.",
    )
    note("2:20--3:15. The training split is class-imbalanced. Eight validation images duplicated training images exactly, so I removed them before selecting checkpoints.")


def methods():
    text("""
    ## Models and augmentation

    The CNN is a compact ResNet-18-style model. The Vision Transformer uses $16\\times16$ patches, six encoder layers, 256-dimensional embeddings, and eight attention heads.

    For C1 and T1, each training image has a 75% chance of receiving one moderate degradation:
    """)
    table(AUGMENTATIONS, caption="Training-only quality-aware augmentation policy")
    note("3:15--4:20. These perturbations simulate quality variation. I did not use flips, large rotations, or crops because the goal was not geometric augmentation.")


def evaluation():
    text("""
    ## Evaluation protocol

    Each selected model sees the same held-out test images under eight conditions: clean, two blur severities, two noise severities, two contrast levels, and low resolution.

    **Primary metric:** AUROC  
    **Secondary metrics:** AUPRC, balanced accuracy, macro-F1, sensitivity, and specificity
    """)
    text("""
    The study uses class-weighted cross-entropy, AdamW, a maximum of 35 epochs, early stopping on validation AUROC, and random seed 17. C0, C1, T0, and T1 selected epochs 13, 4, 10, and 17 respectively.
    """)
    note("4:20--5:10. The held-out test set never controls early stopping. Thresholded metrics use the pre-specified 0.5 probability threshold.")


def clean_results():
    text("## Clean test performance")
    table(CLEAN_RESULTS, caption="Clean held-out test results")
    text("CNN augmentation increased clean AUROC by 0.007. ViT augmentation changed clean AUROC by -0.002. Both changes satisfy the pre-specified 0.02 clean-performance margin.")
    note("5:10--6:10. We do not claim that augmentation improves every metric. The clean AUROC result supports H2 for both models under the chosen reporting margin.")


def robustness_results():
    text("## Robustness under image-quality degradation")
    plot(ROBUSTNESS_CHART)
    table(MEAN_DEGRADED, caption="Mean AUROC across the seven degraded conditions")
    note("6:10--8:20. This is the key result. Augmentation improves the CNN in every degraded condition, most strongly under severe contrast loss: 0.781 to 0.932. The ViT remains relatively robust without augmentation, and its mean robustness does not increase.")


def conclusion():
    text("""
    ## Takeaways and limitations

    - The selected quality-aware policy improved CNN mean degraded AUROC from 0.907 to 0.948 without a clean-AUROC penalty.
    - The same policy did not improve ViT mean degraded AUROC.
    - Severe contrast reduction caused the largest AUROC drop for both architectures.

    These results come from one seed and synthetic corruptions in a pediatric benchmark. They do not establish clinical deployment robustness.
    """)
    text("""
    ### Next steps

    Repeat across seeds, evaluate naturally degraded and external hospital data, and test augmentation policies designed specifically for Vision Transformers.
    """)
    note("8:20--10:00. End with the qualified answer to the research question: quality-aware augmentation can help, but the benefit depends on the architecture and the test shift.")
if __name__ == "__main__":
    main()
