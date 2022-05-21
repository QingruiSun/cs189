import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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
    data_list = [mnist_validate_data, mnist_validate_labels, mnist_training_data, mnist_training_labels,
                 spam_validate_data, spam_validate_labels, spam_training_data, spam_training_labels,
                 cifar10_validate_data, cifar10_validate_labels, cifar10_training_data, cifar10_training_labels]
    return data_list


def train_svm(data, labels, num_list, kernel='linear', C=1.0):
    clf_list = []
    for num in num_list:
        clf = SVC(max_iter=1000000, kernel=kernel, C=C)
        clf.fit(data[:num], np.ravel(labels[:num]))
        clf_list.append(clf)
    return clf_list


def calculate_accuracy(clf, data, labels):
    accuracy = clf.score(data, np.ravel(labels))
    return accuracy


def train_mnist(data_list):
    num_list = [100, 200, 500, 1000, 2000, 5000, 10000]
    clf_list = train_svm(data_list[2], data_list[3], num_list)
    train_accuracy_list = []
    validate_accuracy_list = []
    for i in range(len(num_list)):
        num = num_list[i]
        clf = clf_list[i]
        train_accuracy = calculate_accuracy(clf, data_list[2][:num], data_list[3][:num])
        validate_accuracy = calculate_accuracy(clf, data_list[0], data_list[1])
        train_accuracy_list.append(train_accuracy)
        validate_accuracy_list.append(validate_accuracy)
    plt.plot(num_list, train_accuracy_list, label='mnist training accuracy')
    plt.plot(num_list, validate_accuracy_list, label='mnist validate accuracy')
    plt.xlabel("training nums")
    plt.ylabel("accuracy")
    plt.show()


def train_spam(data_list):
    num_list = [100, 200, 500, 1000, 2000, data_list[6].shaep[0]]
    clf_list = train_svm(data_list[6], data_list[7], num_list)
    train_accuracy_list = []
    validate_accuracy_list = []
    for i in range(len(num_list)):
        num = num_list[i]
        clf = clf_list[i]
        train_accuracy = calculate_accuracy(clf, data_list[6][:num], data_list[7][:num])
        validate_accuracy = calculate_accuracy(clf, data_list[4], data_list[5])
        train_accuracy_list.append(train_accuracy)
        validate_accuracy_list.append(validate_accuracy)
    plt.plot(num_list, train_accuracy_list, label='spam training accuracy')
    plt.plot(num_list, validate_accuracy_list, label='spam validate accuracy')
    plt.xlabel("training nums")
    plt.ylabel("accuracy")
    plt.show()


def train_cifar10(data_list):
    num_list = [100, 200, 500, 1000, 2000, 5000]
    clf_list = train_svm(data_list[10], data_list[11], num_list)
    train_accuracy_list = []
    validate_accuracy_list = []
    for i in range(len(num_list)):
        num = num_list[i]
        clf = clf_list[i]
        train_accuracy = calculate_accuracy(clf, data_list[10][:num], data_list[11][:num])
        validate_accuracy = calculate_accuracy(clf, data_list[8], data_list[9])
        train_accuracy_list.append(train_accuracy)
        validate_accuracy_list.append(validate_accuracy)
    plt.plot(num_list, train_accuracy_list, label='cifar10 training accuracy')
    plt.plot(num_list, validate_accuracy_list, label='cifar10 validate accuracy')
    plt.xlabel("training nums")
    plt.ylabel("accuracy")
    plt.show()



if __name__ == "__main__":
    shuffle()
    data_list = shuffle()
    train_mnist(data_list)

