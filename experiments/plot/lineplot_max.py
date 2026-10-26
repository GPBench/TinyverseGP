from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

D_MIN = 1
D_MAX = 10
Y_MIN = 0
Y_MAX = 60000
STEP = 1
X_TICKS = [n for n in range(D_MIN, D_MAX + 1, STEP)]
#X_TICKS += [30]
Y_LIM = True

plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"font.family": "serif"})
plt.rcParams.update({"mathtext.fontset":"cm"})

data = pd.read_csv('../csv/max_plus_mul_tgp_cgp_1_10.csv')
plt.figure(figsize=(8, 6.5))
sns.set(font_scale=1.1)

fig, ax = plt.subplots(figsize=(8, 6))

# errorbar= ('ci',95),
p = sns.lineplot(
    data=data,
    x="d", y="num_iters",
    hue="model", style="model",
    markers=True,
    errorbar=('ci',95),
    linewidth = 2.3
)

#yscale="log"
p.set(xlabel="$D$", ylabel="Average Number of Iterations", yscale="linear")
p.set(yscale="linear")

#p.set_xlabel(p.get_xlabel(), fontdict={'weight': 'bold'})
#p.set_ylabel(p.get_ylabel(), fontdict={'weight': 'bold'})
#p.set_title(p.get_title(), fontdict={'weight': 'bold'})

p.set_xticks(X_TICKS)
p.set_xticklabels(str(d) for d in X_TICKS)
plt.xlim(D_MIN, D_MAX)
plt.legend(title="")
if Y_LIM:
    plt.ylim(Y_MIN, Y_MAX)

fig.tight_layout()
plt.savefig("max_plus_mul_tgp_cgp_1_10.png", dpi=150)
plt.show()