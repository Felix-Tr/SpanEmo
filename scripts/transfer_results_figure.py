import pandas as pd
import re
from pathlib import Path


drive_ws = Path("/content/drive/MyDrive/temporary/masterthesis_drive")
runs_ws = drive_ws / "models" / "20220607" / "SpaEmo"

results_df = pd.DataFrame(columns=["model", "F1-Macro", "F1-Micro", "JS"])
langs = ["GerSentiment", "German", "English"]
for lang in langs:
    for case in ["pretrained", "base"]:
        for frac in [100, 200, 500, 1000, 5000]:
            name = f"SpanEmo-{lang}-{case}-{frac}"

            file = list((runs_ws / name).glob("*test_result.txt"))[0]
            with open(file) as f:
                result = f.readlines()

            result = [re.sub("\s|\n", "", s) for s in result][:-1]
            result = {s.split(":")[0]:[float(s.split(":")[1])] for s in result}
            result["model"] = [name]
            results_df = pd.concat([results_df, pd.DataFrame(result)])



import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

x = [100, 200, 500, 1000, 5000]

fig, axs = plt.subplots(2, 2, figsize=(21,9), dpi=1000)
for i, lang in zip([0, 0, 1], langs):
    for case in ["pretrained", "base"]:
        for j, metric in enumerate(["F1-Macro", "F1-Micro"]):
            results_temp = results_df[[(lang in row) and (case in row) for row in results_df.model]]
            label = f"{lang}-{case}"
            try:
                axs[i, j].plot(x, results_temp[metric].tolist(), "-o", label=label)
                axs[i, j].set_xlabel('# datapoints')
                axs[i, j].set_ylabel(metric)
                axs[i, j].grid(True)
                axs[i, j].legend()
                axs[i, j].set_title(f"{lang}-{metric}")
            except:
                print("debug")

plt.tight_layout()
# plt.figure(figsize=(21, 9))
plt.savefig(runs_ws / ("Transfer_Results.svg"))
