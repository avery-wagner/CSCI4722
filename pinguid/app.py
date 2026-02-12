"""
* PinguID *
LandingLens Implementation Lab
CSCI 4722 - Computer Vision
Avery Wagner | Due Feb. 12, 2026

This is a simple app using the "model3" deployment of my penguin segmentation project on LandingLens to detect penguins in images.

"""

from PIL import Image
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

from datetime import datetime as Date
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display, Markdown
from dotenv import load_dotenv

load_dotenv()
MIN_HIST_SAMPLES = 10


def load_and_predict(dataset_name, image_paths, predictor, options, num_samples=3):
    print(f"\n> Processing {dataset_name} dataset...")

    if len(image_paths) < num_samples:
        print(f" [!] Warning: Only {len(image_paths)} images available - using all")
        num_samples = len(image_paths)

    rand_idx = np.random.choice(len(image_paths), size=num_samples, replace=False)

    images = []
    predictions = []
    overlays = []

    for i, idx in enumerate(rand_idx):
        print(f" > Image {i+1}/{num_samples}...")
        img = Image.open(image_paths[idx])
        images.append(img)

        pred = predictor.predict(img)
        predictions.extend(pred)

        overlay = overlay_predictions(pred, img, options=options)
        overlays.append(overlay)

    return images, predictions, overlays


def analyze_results(tr_preds, new_preds, threshold=0.5):
    if not tr_preds and not new_preds:
        return None

    all_scores_tr = [p.score for p in tr_preds]
    high_conf_tr = [p for p in tr_preds if p.score >= threshold]
    all_scores_new = [p.score for p in new_preds]
    high_conf_new = [p for p in new_preds if p.score >= threshold]

    t_stat, p_value = stats.ttest_ind(all_scores_tr, all_scores_new)

    stats_tr = {
        "high_confidence_detections": len(high_conf_tr),
        "mean_confidence": np.mean(all_scores_tr),
        "std_confidence": np.std(all_scores_tr),
        "median_confidence": np.median(all_scores_tr),
        "min_confidence": np.min(all_scores_tr),
        "max_confidence": np.max(all_scores_tr),
        "all_scores": all_scores_tr,
        "t_stat": t_stat,
        "p_value": p_value,
    }

    stats_new = {
        "high_confidence_detections": len(high_conf_new),
        "mean_confidence": np.mean(all_scores_new),
        "std_confidence": np.std(all_scores_new),
        "median_confidence": np.median(all_scores_new),
        "min_confidence": np.min(all_scores_new),
        "max_confidence": np.max(all_scores_new),
        "all_scores": all_scores_new,
        "t_stat": t_stat,
        "p_value": p_value,
    }

    return stats_tr, stats_new


def generate_boxplots(stats_train, stats_new, report_dir, threshold):
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [stats_train["all_scores"], stats_new["all_scores"]]
    bp = ax.boxplot(
        data, tick_labels=["Training Dataset", "New Dataset"], patch_artist=True
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")

    ax.axhline(
        threshold,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Threshold: {threshold}",
    )
    ax.set_title("Comparison of Confidence Score Distributions")
    ax.set_ylabel("Confidence Score")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{report_dir}/boxplots.png", dpi=150)
    plt.show()


def generate_histograms(stats_train, stats_new, report_dir, threshold):
    # Confidence distribution comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        stats_train["all_scores"],
        bins=20,
        # density=True,
        alpha=0.7,
        color="blue",
    )

    axes[1].hist(
        stats_new["all_scores"],
        bins=20,
        # density=True,
        alpha=0.7,
        color="green",
    )

    axes[0].axvline(
        stats_train["mean_confidence"],
        color="red",
        linestyle="--",
        label=f"Mean: {stats_train['mean_confidence']:.3f}",
    )
    axes[0].axvline(
        threshold,
        color="orange",
        linestyle=":",
        label=f"Threshold: {threshold}",
    )
    axes[0].set_xlabel("Confidence Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Training Dataset - Confidence Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axvline(
        stats_new["mean_confidence"],
        color="red",
        linestyle="--",
        label=f"Mean: {stats_new['mean_confidence']:.3f}",
    )
    axes[1].axvline(
        threshold,
        color="orange",
        linestyle=":",
        label=f"Threshold: {threshold}",
    )
    axes[1].set_xlabel("Confidence Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("New Dataset - Confidence Distribution")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{report_dir}/histograms.png", dpi=150)
    plt.show()


def generate_report(stats_train, stats_new, report_dir, threshold, N):
    if N >= MIN_HIST_SAMPLES:
        generate_histograms(stats_train, stats_new, report_dir, threshold)
    generate_boxplots(stats_train, stats_new, report_dir, threshold)

    date = Date.fromisoformat(
        str.split(str(report_dir), "reports/report-")[1].rstrip("/")
    ).ctime()
    results_output = f"""
#  🐧 PinguID Report - {date}

------

![result boxplots](./boxplots.png)
{f"![result histogram plots](./histograms.png)" if N >= MIN_HIST_SAMPLES else ""}

| Statistic | Training Dataset Results | New Dataset Results |
| --- | --- | --- |
| **# Confidence > {threshold}** | {stats_train['high_confidence_detections']} | {stats_new['high_confidence_detections']} |
| **Mean Confidence** | {stats_train['mean_confidence']:.5f} | {stats_new['mean_confidence']:.5f} |
| **Std Dev Confidence** | {stats_train['std_confidence']:.5f} | {stats_new['std_confidence']:.5f} |
| **Median Confidence** | {stats_train['median_confidence']:.5f} | {stats_new['median_confidence']:.5f} |
| **Confidence Range** | [{stats_train['min_confidence']:.5f}, {stats_train['max_confidence']:.5f}] | [{stats_new['min_confidence']:.5f}, {stats_new['max_confidence']:.5f}] |

| | | | |
|--|--|--|--|
| **T-test** ($\\alpha = 0.05$) | **t-stat** = {stats_train['t_stat']:.3f}  | **p-value** = {stats_train['p_value']:.3f} | $\implies$ {"Statistically significant difference ($p < 0.05$)" if stats_train['p_value'] < 0.05 else "No significant difference ($p \geq 0.05$)"} |
| | | | |
    """

    with open(f"{report_dir}/report.md", "w", encoding="utf-8") as f:
        f.write(results_output)


def main():
    print("\n>>> 🐧 STARTING PINGUID 🐧 <<<")

    ENDPOINT_ID = os.getenv("ENDPOINT_ID")
    API_KEY = os.getenv("API_KEY")
    CONFIDENCE_THRESHOLD = 0.995
    NUM_SAMPLES = 3
    DISPLAY_PREDS = True if NUM_SAMPLES <= 3 else False

    report_parent_dir = Path(f"reports/")
    report_dir = Path(f"./reports/report-{Date.now().strftime('%Y-%m-%d-%H:%M:%S')}/")
    new_pred_dir = Path(report_dir / "new_predictions")
    train_pred_dir = Path(report_dir / "training_predictions")

    report_parent_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    new_pred_dir.mkdir(exist_ok=True)
    train_pred_dir.mkdir(exist_ok=True)

    predictor = Predictor(ENDPOINT_ID, api_key=API_KEY)
    color_map = {"Penguin": "blue", "Seal": "magenta"}
    options = {"color_map": color_map}

    # Load all images
    pics_training = list(Path("./penguin-pics").glob("*.jp*g"))
    pics_new = list(Path("./penguin-pics-2").glob("*.jp*g"))

    # Run predictions on both datasets
    imgs_train, preds_train, overlays_train = load_and_predict(
        "Training", pics_training, predictor, options, NUM_SAMPLES
    )
    imgs_new, preds_new, overlays_new = load_and_predict(
        "New", pics_new, predictor, options, NUM_SAMPLES
    )

    print("\n> Analyzing results + creating report...")

    stats_train, stats_new = analyze_results(
        preds_train, preds_new, CONFIDENCE_THRESHOLD
    )
    generate_report(
        stats_train, stats_new, report_dir, CONFIDENCE_THRESHOLD, NUM_SAMPLES
    )

    # Save + Display prediction overlays if enabled
    print("\n> Saving prediction overlays...")

    for i, overlay in enumerate(overlays_train):
        if DISPLAY_PREDS:
            print(f" > Displaying Training Image {i+1}:")
            overlay.show()
        overlay.save(train_pred_dir / f"tr_pred_{i+1}.png")

    for i, overlay in enumerate(overlays_new):
        if DISPLAY_PREDS:
            print(f" > Displaying New Image {i+1}:")
            overlay.show()
        overlay.save(new_pred_dir / f"new_pred_{i+1}.png")

    print(f"\n✓ Overlays saved to {report_dir}")
    print("\n>>> 🐧 DONE 🐧 <<<\n")


if __name__ == "__main__":
    main()
