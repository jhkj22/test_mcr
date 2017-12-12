import sys, os
sys.path.append('/home/hayato/machine_learning/deep-learning-from-scratch')
from dataset.mnist import load_mnist
import numpy as np

def get_image(num):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)
    img = np.uint8(x_train[num])
    img = img.reshape(28, 28)
    return img

if __name__ == '__main__':
    img = get_image(0)
    from PIL import Image
    img = Image.fromarray(img)
    img.save('xt_0_5.jpg')
    

