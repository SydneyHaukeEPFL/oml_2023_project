import pandas as pd
import yaml
import matplotlib.pyplot as plt


def load_results(results_path: str):
    with open(results_path, "r", encoding="utf-8") as results_file:
        results = yaml.safe_load(results_file)

    # Separate config and results
    config = results["Config"]
    results = results["Epoch"]

    # Get all metric names
    metric_names = set()
    for epoch in results:
        for name in results[epoch]:
            metric_names.add(name)

    # Get metrics values epoch by epoch
    metric_values = {name: [] for name in metric_names}
    metric_values["epoch"] = []
    for epoch in results:
        metric_values["epoch"].append(epoch)
        for name in metric_names:
            if name in results[epoch]:
                metric_values[name].append(results[epoch][name])
            else:
                metric_values[name].append(None)

    # Store results in a Dataframe
    results = pd.DataFrame(data=metric_values)

    return config, results


if __name__ == "__main__":
    ax = plt.gca()
    config, results = load_results("results/wine_mlp_zero_order_108.yaml")
    print(config["u"], config["lr"], config["batch_size"])
    # results.plot(x="epoch", y="Val loss", ax=ax, label="ZO-SGD")
    results.plot(x="epoch", y="Val loss", ax=ax, label="ZO-SGD")
    config, results = load_results("results/wine_mlp_sgd_174.yaml")
    print(config["u"], config["lr"], config["batch_size"])
    # results.plot(x="epoch", y="Val loss", ax=ax, label="SGD")
    results.plot(x="epoch", y="Val loss", ax=ax, label="SGD")
    ax.set_title("Comparison of valdation MSE during training")
    ax.set_ylabel("Val MSE")
    ax.set_xlim(0, 80)
    plt.savefig("MSE MLP", dpi=300)
    plt.show()
