import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams['font.size'] = 34

df = pd.read_csv('New_CNN.csv')

epochs = pd.Series(range(1, 31))
val_acc = df['Val Accuracy'].values.reshape(10, 30)
train_acc = df['Train Accu'].values.reshape(10, 30)
val_loss = df['Val Loss'].values.reshape(10, 30)
train_loss = df['Train Loss'].values.reshape(10, 30)

val_acc_mean = val_acc.mean(axis=0)
val_acc_std = val_acc.std(axis=0)
val_loss_mean = val_loss.mean(axis=0)
val_loss_std = val_loss.std(axis=0)

fig, ax1 = plt.subplots(figsize=(12,8))

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)', color='tab:red')
ax1.plot(epochs, val_acc_mean, color='r', marker='o', linestyle='-', label='Val Accuracy', markersize=3)
ax1.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, color='r', alpha=0.2, label='Std Deviation')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True)


ax2 = ax1.twinx()
ax2.set_ylabel('Validation Loss', color='tab:blue')
ax2.plot(epochs, val_loss_mean, color='b', marker='s', linestyle='--', label='Val Loss', markersize=3)
ax2.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, color='blue', alpha=0.2, label='Std Deviation')
ax2.tick_params(axis='y', labelcolor='tab:blue')


# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1), framealpha=1.0)

ax1.set_xlim([min(epochs) - 1, max(epochs) + 1])
ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))

ax1.xaxis.set_major_locator(MultipleLocator(5))


# ax1.yaxis.set_major_locator(MaxNLocator(nbins=10))

ax1.yaxis.set_major_locator(MultipleLocator(2.5))


ax2.yaxis.set_major_locator(MultipleLocator(0.25))

ax1.grid(True, linestyle='-.')
ax2.grid(True, linestyle='-.')

ax1.set_ylim(65, 85)
ax2.set_ylim(0, 2)

plt.tight_layout()

plt.savefig('./results/New_CNN.png', dpi=500)
plt.show()



