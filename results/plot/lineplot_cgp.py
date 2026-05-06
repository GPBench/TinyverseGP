import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"mathtext.fontset":"cm"})

D_MIN = 1
D_MAX = 30

Y_MIN = 0
Y_MAX = 100000

Y_LIM = False
X_LIM = False

STEP = 5
X_TICKS = [n for n in range(D_MIN-1, D_MAX + 1, STEP)]
X_TICKS += [D_MAX]

data = pd.read_csv('max_plus_mul_log_cgp_sam_prob_1_30.csv')

fig, ax = plt.subplots(figsize=(8, 6))

p = sns.lineplot(
    data=data,
    x="d", y="num_iters",
    hue="model", style="model",
    markers=True, dashes=False,
    ax=ax
)

p.set(xlabel="$D$", ylabel="Average Number of Iterations", yscale="linear")
p.set_xticks(X_TICKS)
p.set_xticklabels(str(d) for d in X_TICKS)


if X_LIM:
    plt.xlim(D_MIN, D_MAX)

if Y_LIM:
    plt.ylim(Y_MIN, Y_MAX)

fig.tight_layout()
plt.savefig("max_plus_mul_log_cgp_sam_prob_1_30.svg")
plt.show()
