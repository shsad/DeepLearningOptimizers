import matplotlib.pyplot as plt

train_loss_sgd = [0.6160, 0.3285, 0.2863, 0.2562, 0.2297, 0.2082, 0.1894, 0.1736, 0.1598, 0.1476, 0.1368, 0.1278, 0.1191, 0.1122, 0.1057, 0.0999, 0.0947, 0.0899, 0.0852, 0.0817]
val_loss_sgd = [0.3801, 0.3262, 0.2819, 0.2637, 0.2334, 0.2146, 0.2016, 0.1889, 0.1735, 0.1708, 0.1607, 0.1504, 0.1450, 0.1370, 0.1343, 0.1308, 0.1250, 0.1215, 0.1176, 0.1186]

train_loss_msgd = [0.3386, 0.1689, 0.1249, 0.1039, 0.0896, 0.0761, 0.0698, 0.0615, 0.0546, 0.0503, 0.0458, 0.0401, 0.0353, 0.0360, 0.0308, 0.0258, 0.0245, 0.0215, 0.0215, 0.0190]
val_loss_msgd = [0.2263, 0.1654, 0.1793, 0.1152, 0.1123, 0.1479, 0.1116, 0.1078, 0.1099, 0.1075, 0.1268, 0.1080, 0.1124, 0.1183, 0.1074, 0.1088, 0.1324, 0.1040, 0.1090, 0.1074]

train_loss_nesterov = [
    0.3388, 0.1659, 0.1244, 0.0988, 0.0849, 0.0733, 0.0656, 0.0579, 0.0504, 0.0449,
    0.0427, 0.0366, 0.0324, 0.0302, 0.0277, 0.0255, 0.0210, 0.0189, 0.0163, 0.0153
 ]
val_loss_nesterov = [
    0.2052, 0.1602, 0.1349, 0.1184, 0.1139, 0.1344, 0.1217, 0.1055, 0.1014, 0.1121,
    0.1010, 0.1058, 0.1033, 0.1149, 0.1215, 0.1118, 0.1019, 0.1068, 0.0974, 0.0972
]

train_loss_adagrad = [0.3917, 0.2626, 0.2291, 0.2080, 0.1913, 0.1781, 0.1672, 0.1590, 0.1517, 0.1452, 0.1397, 0.1346, 0.1300, 0.1259, 0.1220, 0.1188, 0.1155, 0.1128, 0.1100, 0.1074]
val_loss_adagrad = [0.2906, 0.2488, 0.2260, 0.2097, 0.1939, 0.1839, 0.1749, 0.1686, 0.1619, 0.1576, 0.1534, 0.1539, 0.1467, 0.1434, 0.1393, 0.1377, 0.1345, 0.1334, 0.1317, 0.1317]

train_loss_adam = [0.3717, 0.1889, 0.1396, 0.1137, 0.1009, 0.0875, 0.0775, 0.0713, 0.0653, 0.0603, 0.0540, 0.0520, 0.0463, 0.0440, 0.0406, 0.0387, 0.0364, 0.0365, 0.0327, 0.0323]
val_loss_adam = [0.2171, 0.1539, 0.1564, 0.1391, 0.1138, 0.1116, 0.1151, 0.1105, 0.1270, 0.1099, 0.1308, 0.1335, 0.1293, 0.1182, 0.1273, 0.1296, 0.1309, 0.1661, 0.1474, 0.1467]


# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_loss_sgd, label="MySGD", color='blue')
plt.plot(train_loss_msgd, label="MyMomentumSGD", color='red')
plt.plot(train_loss_nesterov, label="MyNesterovMomentumSGD", color='green')
plt.plot(train_loss_adagrad, label="MyAdaGrad", color='orange')
plt.plot(train_loss_adam, label="MyAdam", color='purple')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_loss_sgd, label="MySGD", color='blue')
plt.plot(val_loss_msgd, label="MyMomentumSGD", color='red')
plt.plot(val_loss_nesterov, label="MyNesterovMomentumSGD", color='green')
plt.plot(val_loss_adagrad, label="MyAdaGrad", color='orange')
plt.plot(val_loss_adam, label="MyAdam", color='purple')
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()