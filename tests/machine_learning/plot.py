import pandas as pd
import numpy as np
import matplotlib as plt

import matplotlib
matplotlib.use("TkAgg")  # Do this before importing pyplot!
import matplotlib.pyplot as plt

def show_plot():
    ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))

    # ts = ts.cumsum()
    # ts.plot()

    # plt.imshow(ts.reshape((28, 28)))
    plt.plot([1, 2, 3, 4])
    plt.show()


if __name__ == '__main__':
    show_plot()