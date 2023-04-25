import json
import matplotlib.pyplot as plt
import sys


def parse(filename):
    steps, accuracy = [], []

    with open (filename, 'r') as f:
        for line in f.readlines():
            entry = json.loads(line)
            if (r"val/prec@1") in entry:
                steps.append(entry["step"])
                accuracy.append(entry[r"val/prec@1"])

    return steps, accuracy


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "big_vision_metrics.txt"

    # current model, blue
    x, y = parse(filename)

    # baseline, orange
    x_b, y_b = parse("04-26_0003.txt")

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x_b, y_b)

    plt.savefig(f"{filename.split('.')[0]}-compare.png")
