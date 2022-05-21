import numpy as np
import scipy.io as sio

mnist_data_path = "../data/mnist_data.mat"
spam_data_path = "../data/spam_data.mat"
cifar10_data_path = "../data/cifar10_data.mat"


def shuffle_help(data, labels):
    length = data.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices)
    return data[indices], labels[indices]


def shuffle():
    #load data
    mnist_data = sio.loadmat(mnist_data_path)
    mnist_training_data = mnist_data['training_data']
    mnist_training_labels = mnist_data['training_labels']
    spam_data = sio.loadmat(spam_data_path)
    spam_training_data = spam_data['training_data']
    spam_training_labels = spam_data['training_labels']
    cifar10_data = sio.loadmat(cifar10_data_path)
    cifar10_training_data = cifar10_data['training_data']
    cifar10_training_labels = cifar10_data['training_labels']
    #shuffle data
    mnist_training_data, mnist_training_labels = shuffle_help(mnist_training_data, mnist_training_labels)
    spam_training_data, spam_training_labels = shuffle_help(spam_training_data, spam_training_labels)
    cifar10_training_data, cifar10_training_labels = shuffle_help(cifar10_training_data, cifar10_training_labels)
    #partition data
    mnist_validate_data, mnist_validate_labels = mnist_training_data[:10000], mnist_training_labels[:10000]
    mnist_training_data, mnist_training_labels = mnist_training_data[10000:], mnist_training_labels[10000:]
    spam_num = spam_training_data.shape[0]
    spam_validate_data, spam_validate_labels = spam_training_data[:int(0.2 * spam_num)], spam_training_labels[:int(0.2 * spam_num)]
    spam_training_data, spam_training_labels = spam_training_data[int(0.2 * spam_num):], spam_training_labels[int(0.2 * spam_num):]
    cifar10_validate_data, cifar10_validate_labels = cifar10_training_data[:5000], cifar10_training_labels[:5000]
    cifar10_training_data, cifar10_training_labels = cifar10_training_data[5000:], cifar10_training_labels[5000:]

def train_svm():
    




if __name__ == "__main__":
    shuffle()
