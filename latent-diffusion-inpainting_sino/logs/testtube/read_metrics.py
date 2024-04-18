import pandas
import matplotlib.pyplot as plt
import numpy as np

epoch = 'upto_ep430/'
file_path = 'version_1/metrics.csv'
metrics_all = pandas.read_csv(file_path)
print(metrics_all.shape)
print(metrics_all)
loss_LDM = metrics_all['train/loss_simple_step'].dropna()
loss_vlb = metrics_all['train/loss_vlb_step'].dropna()
loss_step = metrics_all['train/loss_step'].dropna()

if (loss_LDM == loss_step).all():
    print('loss_simple_step and loss_step are equal !!!')

'''
plt.plot(loss_step)
plt.xlabel('Training Steps')
plt.ylabel('Total Loss / Step')
plt.title('Total Loss over Training Steps')
plt.savefig('metrics_plots/' + epoch + 'Total_loss_steps.png')

plt.plot(loss_vlb)
plt.xlabel('Training Steps')
plt.ylabel('Autoencoder Loss / Step')
plt.title('Autoencoder Loss over Training Steps')
plt.savefig('metrics_plots/' + epoch + 'AE_loss_steps.png')
'''

loss_LDM = loss_LDM.to_numpy()
loss_vlb = loss_vlb.to_numpy()

L = loss_LDM.shape[0]
print('L', L)
loss_LDM_mv = np.zeros(int(np.ceil(L/100)))
loss_vlb_mv = np.zeros(int(np.ceil(L/100)))
print(loss_LDM_mv.shape)
print(loss_vlb_mv.shape)

jj = 0
for ii in range(0, L, 100):
    print('ii, jj', ii, jj)
    loss_LDM_i = loss_LDM[ii : ii+100]
    loss_vlb_i = loss_vlb[ii : ii+100]
    print(loss_LDM_i.shape)
    print(loss_vlb_i.shape)
    loss_LDM_mv[jj] = np.mean(loss_LDM_i)
    loss_vlb_mv[jj] = np.mean(loss_vlb_i)
    jj += 1

print('end of steps !!!')

'''
plt.plot(loss_LDM_mv)
plt.xlabel('Training Steps')
plt.ylabel('Total Loss / Step')
plt.title('Total Loss over Training Steps')
plt.savefig('metrics_plots/' + epoch + 'Tot_rolling_loss_steps.png')
'''

'''
plt.plot(loss_vlb_mv)
plt.xlabel('Training Steps')
plt.ylabel('Autoencoder Loss / Step')
plt.title('Autoencoder Loss over Training Steps')
plt.savefig('metrics_plots/' + epoch + 'AE_rolling_loss_steps.png')
'''
