import numpy as np
import matplotlib.pyplot as plt

alphas = np.arange(6, 50, 12)
dropout_lst = np.arange(0, 0.50, 0.10)

for alpha in alphas:
    for dropout in dropout_lst:
        ind_test = np.loadtxt("losses/ind_test_u{}_d{}.txt".format(alpha, dropout))
        ind_train = np.loadtxt("losses/ind_train_u{}_d{}.txt".format(alpha, dropout))
        jnt_test = np.loadtxt("losses/jnt_test_u{}_d{}.txt".format(alpha, dropout))
        jnt_train = np.loadtxt("losses/jnt_train_u{}_d{}.txt".format(alpha, dropout))
        
        plt.figure(figsize=(8,6))
        plt.plot((ind_train), "r-")
        plt.plot((ind_test), "r--")
        plt.plot((jnt_train), "b-")
        plt.plot((jnt_test), "b--")

        plt.legend(["Training (Independent)","Testing (Independent)","Training (Joint)", "Testing (Joint)"])
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        
        plt.savefig("learning_curve_u{}_d{}.pdf")