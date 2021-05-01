import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.genfromtxt('loss.csv', delimiter=',')
    data = data*(10**5)
    epochs = list(range(0, 225))
    plt.plot(epochs, data[0, :], label="Training Loss")
    plt.plot(epochs, data[1, :], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel('Normalized Loss')
    plt.ylim(bottom=0)
    plt.legend()
    #plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

if __name__ == '__main__':
    main()
