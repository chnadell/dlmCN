import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid

model_name = '20180705_180250'
pred_file = os.path.join(os.path.dirname(__file__), 'data', 'test_pred_{}.csv'.format(model_name))
truth_file = os.path.join(os.path.dirname(__file__), 'data', 'test_truth.csv')

pred = np.loadtxt(pred_file, delimiter=' ')
truth = np.loadtxt(truth_file, delimiter=' ')

for fig_cnt in range(13):
    fig = plt.figure(figsize=(12, 8))
    grid = Grid(fig, rect=111, nrows_ncols=(4, 4), axes_pad=0.25, label_mode='L')
    for i, ax in enumerate(grid):
        try:
            ax.plot(truth[fig_cnt*16+i, :], label='truth')
            ax.plot(pred[fig_cnt*16+i, :], label='pred')
            ax.legend()
            mse = np.mean(np.absolute(truth[fig_cnt*16+i, :] - pred[fig_cnt*16+i, :]))
            plt.text(0.6, 0.4, 'MAE={:.3f}'.format(mse), ha='center', va='center', transform=ax.transAxes)
        except IndexError:
            pass
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figs', 'prediction_plot_{}_{}.png'.format(model_name,
                                                                                                   fig_cnt)))
    plt.close(fig)
