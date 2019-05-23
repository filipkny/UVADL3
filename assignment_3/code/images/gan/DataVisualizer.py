import pickle
import matplotlib.pyplot as plt
filename = 'gan_losses_2019-05-18 21:32'
with open (filename, 'rb') as fp:
    itemlist = pickle.load(fp)

[d_losses, g_losses, avg_g_losses, avg_d_losses] = itemlist
# [d_losses, g_losses] = itemlist
lim = 5500
plt.plot(avg_g_losses,label = "discriminator")
plt.plot(avg_d_losses,label = "generator")
plt.legend()
plt.grid(True)
plt.savefig("plot_"+filename+".png")
