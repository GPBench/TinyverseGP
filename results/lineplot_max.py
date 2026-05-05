import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

D_MIN = 1
D_MAX = 30

Y_MIN = 0
Y_MAX = 100000

Y_LIM = False
X_LIM = True

STEP = 1
X_TICKS = [n for n in range(D_MIN, D_MAX + 1, STEP)]
X_TICKS += [D_MAX]

data = pd.read_csv('../examples/analysis/max_plus_mul_log_cgp_sam_prob.csv')
plt.figure(figsize=(8, 6))

p = sns.lineplot(
    data=data,
    x="d", y="num_iters",
    hue="model", style="model",
    markers=True
)

p.set(xlabel='n', ylabel='Average Number of Iterations', yscale="linear")
p.set_xticks(X_TICKS)
p.set_xticklabels(str(d) for d in X_TICKS)


if X_LIM:
    plt.xlim(D_MIN, D_MAX)

if Y_LIM:
    plt.ylim(Y_MIN, Y_MAX)

plt.savefig("max_plot.svg", dpi=150)
plt.show()