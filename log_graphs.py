import matplotlib.pyplot as plt
from logs_utils import preprocess_data_to_dataframe


file_path = "1-log_1000epo_50emb_0.001.txt"
df = preprocess_data_to_dataframe(file_path)


plt.figure(figsize=(10, 10))

ax1 = plt.subplot(2, 1, 1)
plt.plot(
    df["epoch"], df["val_ndcg"], c="blue", label="val_ndcg", linestyle="-"
)  # Solid line
plt.plot(
    df["epoch"], df["test_ndcg"], c="blue", label="test_ndcg", linestyle="--"
)  # Dashed line
plt.title("NDCG Scores by Epoch")
plt.ylabel("NDCG Score")
plt.legend()
plt.grid(True)


max_val_ndcg_epoch = df.loc[df["val_ndcg"].idxmax(), "epoch"]
max_val_ndcg_value = df["val_ndcg"].max()
plt.axvline(x=max_val_ndcg_epoch, color="blue", linestyle=":", linewidth=1)
plt.text(
    max_val_ndcg_epoch,
    max_val_ndcg_value,
    f"Epoch {max_val_ndcg_epoch}\nNDCG {max_val_ndcg_value:.2f}",
    color="blue",
    verticalalignment="bottom",
    horizontalalignment="right",
)

max_test_ndcg_epoch = df.loc[df["test_ndcg"].idxmax(), "epoch"]
max_test_ndcg_value = df["test_ndcg"].max()
plt.axvline(x=max_test_ndcg_epoch, color="blue", linestyle="--", linewidth=1)
plt.text(
    max_test_ndcg_epoch,
    max_test_ndcg_value,
    f"Epoch {max_test_ndcg_epoch}\nNDCG {max_test_ndcg_value:.2f}",
    color="blue",
    verticalalignment="bottom",
    horizontalalignment="right",
)


ax2 = plt.subplot(2, 1, 2)
plt.plot(
    df["epoch"], df["val_hr"], c="green", label="val_hr", linestyle="-"
)  # Solid line
plt.plot(
    df["epoch"], df["test_hr"], c="green", label="test_hr", linestyle="--"
)  # Dashed line
plt.title("HR Scores by Epoch")
plt.ylabel("HR Score")
plt.legend()
plt.grid(True)


max_val_hr_epoch = df.loc[df["val_hr"].idxmax(), "epoch"]
max_val_hr_value = df["val_hr"].max()
plt.axvline(x=max_val_hr_epoch, color="green", linestyle=":", linewidth=1)
plt.text(
    max_val_hr_epoch,
    max_val_hr_value,
    f"Epoch {max_val_hr_epoch}\nHR {max_val_hr_value:.2f}",
    color="green",
    verticalalignment="bottom",
    horizontalalignment="right",
)

max_test_hr_epoch = df.loc[df["test_hr"].idxmax(), "epoch"]
max_test_hr_value = df["test_hr"].max()
plt.axvline(x=max_test_hr_epoch, color="green", linestyle="--", linewidth=1)
plt.text(
    max_test_hr_epoch,
    max_test_hr_value,
    f"Epoch {max_test_hr_epoch}\nHR {max_test_hr_value:.2f}",
    color="green",
    verticalalignment="bottom",
    horizontalalignment="right",
)

plt.savefig("../logs/plots/plot_1.png", dpi=300)

# plt.show()
