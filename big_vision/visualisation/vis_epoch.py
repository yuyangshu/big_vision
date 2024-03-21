import json
import math
import matplotlib.pyplot as plt
import sys


def parse(filename):
    steps, accuracy = [], []

    with open (filename, 'r') as f:
        for line in f.readlines():
            entry = json.loads(line)
            if (r"val/prec@1") in entry:
                step = entry["step"]
                epoch = math.floor(step / 111477 * 90)
                steps.append(epoch)
                accuracy.append(entry[r"val/prec@1"])

    return steps, accuracy


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "big_vision_metrics.txt"

    # current model, blue
    x, y = parse(filename)

    # baseline, orange
    x_b, y_b = parse("07-05_0111.txt")

    fig, ax = plt.subplots()
    ax.plot(x, y, label="Retina ViT")
    ax.plot(x_b, y_b, label="ViT")

    plt.xlabel("Epoch")
    plt.ylabel("Top-1 prediction accuracy")
    plt.xticks(range(0, 91, 10))
    plt.legend(loc="lower right")

    plt.savefig(f"{filename.split('.')[0]}-epoch.png")
