import torch
import numpy as np
import matplotlib.pyplot as plt

from a3_gan_template import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_file = "gen_2019-05-18 21:32"
state_dict = torch.load(model_file)
model = Generator()
model = model.to(device)
model.load_state_dict(state_dict)
model.eval()
step_amount = 9
z = torch.zeros((100, step_amount))
z: torch.Tensor

# convert
steps = torch.FloatTensor((np.linspace(-0.8, 0.2, step_amount)))
z = z.add_(steps).t().to(device)

# generate
pics = model.forward(z)

# plot
for i, pic in enumerate(pics):
    plt.subplot(1, step_amount, i + 1)
    picture = pic.detach().cpu().numpy().reshape((28, 28))
    fig = plt.imshow(picture * -1, cmap='Greys', interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)



plt.show()