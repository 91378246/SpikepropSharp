import sys
import json
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if __name__ == "__main__":    
    file_path = sys.argv[1]
    file = open(file_path)
    data = json.load(file)

    t = np.arange(0, len(data["EegRaw"]), 1)

    # Plot graph with 2 y axes
    fig, ax1 = plt.subplots()

    # Bars
    # Labels raw
    ax1.bar(data["EegLabelsRaw"], len(data["EegLabelsRaw"]) * [1], width=3, color="r")
    # Spikes pos
    spikes_pos = [float(k) for k in data["EcgSignalSpikesTrain"] if data["EcgSignalSpikesTrain"][k]]
    ax1.bar(spikes_pos, len(spikes_pos) * [1], width=3, color="g")
    # Spikes neg
    spikes_neg = [float(k) for k in data["EcgSignalSpikesTrain"] if not data["EcgSignalSpikesTrain"][k]]
    ax1.bar(spikes_neg, len(spikes_neg) * [-1], width=3, color="g")
    # Predictions
    predictionSpan = data["Predictions"][0]["TEnd"]
    predictions_true = [p["TStart"] for p in data["Predictions"] if p["PredictionResult"]]
    ax1.bar(predictions_true, len(predictions_true) * [-.25], width=predictionSpan, color="b")

    ax1.set_xlabel("t")
    ax1.set_ylabel("Spikes", color="r")
    [tl.set_color("r") for tl in ax1.get_yticklabels()]

    # EEG Raw
    ax2 = ax1.twinx()
    ax2.plot(t, data["EegRaw"], label="EEG Raw", color="k", alpha=.25)
    ax2.set_ylabel("Voltage", color="k")
    [tl.set_color("k") for tl in ax2.get_yticklabels()]

    custom_lines = [Line2D([0], [0], color="k", lw=4),
                    Line2D([0], [0], color="r", lw=4),
                    Line2D([0], [0], color="g", lw=4),
                    Line2D([0], [0], color="b", lw=4)]
    plt.legend(custom_lines, ["EEG raw", "Labels", "Spikes pos/neg", "Prediction"])
    plt.show()

    file.close()
