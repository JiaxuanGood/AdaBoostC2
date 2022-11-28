from turtle import color
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("font", family='MicroSoft YaHei')

X_label = ["0.001", "0.005", "0.01", "0.05", "0.1"]
# delta = [0.001, 0.005, 0.01, 0.05, 0.1]
tick_X = np.arange(5)

hamming = [0.115212381,0.109786821,0.107387109,0.109856551,0.113086143]
ranking = [0.158981807,0.156560679,0.152916133,0.154202279,0.152572202]
num_rounds = [3.373803795,2.5934872,2.284946591,1.650533351,1.373628131]
tick_Y1 = [0.105, 0.110, 0.115, 0.120]
Y1_label = ['0.104', '0.108', '0.112', '0.116']
# tick_Y2 = [0, 1, 2, 3, 4]
# Y2_label = ['','1', '2', '3', '4']
tick_Y2 = [1, 1.5, 2, 2.5, 3, 3.5]
Y2_label = ['1','', '2','', '3','']

fig, ax1 = plt.subplots(figsize=(4, 1.8))
# plt.xticks(rotation=45)

ax1.plot(X_label, hamming, color="#c82423", label="Hamming Loss")#+r'$\downarrow$'
# ax1.plot(X, ranking, color="#2878b5", label="Ranking Loss"+r'$\downarrow$')
# ax1.set_xlabel(r'$\delta$')
ax1.set_yticks(tick_Y1)
ax1.set_yticklabels(Y1_label)
ax1.set_ylabel("Hamming Loss", color="#c82423")

ax2 = ax1.twinx()
ax2.plot(tick_X, num_rounds, color="k", marker='o')
ax2.plot(X_label, num_rounds, color="k", label="Iteration rounds")
ax2.set_yticks(tick_Y2)
ax2.set_yticklabels(Y2_label)
ax2.set_ylabel("Iteration rounds")

fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
# plt.show()
plt.savefig('box/parameter2', bbox_inches='tight', dpi=1000)
