import matplotlib.pyplot as plt
import torchvision
import numpy as np


def vizualization(epoch, sample_gen_imgs_in_train):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(f"Generate Images at epoch {epoch}")
    plt.imshow(np.transpose(torchvision.utils.make_grid(sample_gen_imgs_in_train,
                                                        padding=2, normalize=True),
                            (1, 2, 0)))
    plt.show()