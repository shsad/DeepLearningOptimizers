import matplotlib.pyplot as plt

train_loss_sgd = [0.6160, 0.3285, 0.2863, 0.2562, 0.2297, 0.2082, 0.1894, 0.1736, 0.1598, 0.1476, 0.1368, 0.1278, 0.1191, 0.1122, 0.1057, 0.0999, 0.0947, 0.0899, 0.0852, 0.0817]
val_loss_sgd = [0.3801, 0.3262, 0.2819, 0.2637, 0.2334, 0.2146, 0.2016, 0.1889, 0.1735, 0.1708, 0.1607, 0.1504, 0.1450, 0.1370, 0.1343, 0.1308, 0.1250, 0.1215, 0.1176, 0.1186]
train_loss_msgd = [0.3386, 0.1689, 0.1249, 0.1039, 0.0896, 0.0761, 0.0698, 0.0615, 0.0546, 0.0503, 0.0458, 0.0401, 0.0353, 0.0360, 0.0308, 0.0258, 0.0245, 0.0215, 0.0215, 0.0190]
val_loss_msgd = [0.2263, 0.1654, 0.1793, 0.1152, 0.1123, 0.1479, 0.1116, 0.1078, 0.1099, 0.1075, 0.1268, 0.1080, 0.1124, 0.1183, 0.1074, 0.1088, 0.1324, 0.1040, 0.1090, 0.1074]

# Plot the results of experiments and compare different optimizers

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss_sgd, label="MySGD", color='blue')
plt.plot(train_loss_msgd, label="MyMomentumSGD", color='red')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_loss_sgd, label="MySGD", color='blue')
plt.plot(val_loss_msgd, label="MyMomentumSGD", color='red')
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()