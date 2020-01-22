import pickle
from matplotlib import pyplot as plt
import numpy as np

history_file = "1579193916-history.pkl"
delta = 250

file = open(history_file, 'rb')
history = pickle.load(file)
file.close()

acc_raw = np.stack(history['D_acc'])

acc = np.mean(acc_raw.reshape(-1, delta), axis=1)

print(history.keys())

plt.xlabel('Epoch collection', fontsize=18)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Discriminator average accuracy per 100 epochs')

plt.plot(acc)
plt.show()
plt.clf()

d_loss_raw = np.stack(history['D_loss'])
g_loss_raw = np.stack(history['G_loss'])

d_loss = np.mean(d_loss_raw.reshape(-1, delta), axis=1)
g_loss = np.mean(g_loss_raw.reshape(-1, delta), axis=1)

plt.xlabel('Epoch collection', fontsize=18)
plt.ylabel('Loss', fontsize=16)
plt.title('Discriminator/Generator loss average per 100 epochs')

plt.plot(d_loss, label='D loss')
plt.plot(g_loss, label='G loss')

plt.legend()
plt.show()