import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import *

from utils.colors import COLORS, tableau_10_medium


def get_avg_xy(path_prefix, lower_bound, upper_bound, total_num):
    files = glob.glob(path_prefix + '*.csv')

    x_list, y_list = [], []
    for file in files:
        df = pd.read_csv(file)
        x = [i / total_num for i in df.num_total_labeled]

        # y = []
        # for value in df.eval_acc.values:
        #     if isinstance(value, tuple):
        #         y.append(value[0])
        #     else:
        #         y.append(value if value < upper_bound else upper_bound)
        y = [float(v) if v < upper_bound else upper_bound for v in df.eval_acc.values]
        if y[0] != lower_bound:
            y[0] = lower_bound

        if len(y) < 47:
            for num in range(1, 47 - len(y) + 1):
                insert_x = (df.num_total_labeled.iloc[-1] + num * 200) / total_num
                insert_x = 1 if insert_x > 1 else insert_x
                x.append(insert_x)
                insert_y = y[-1] + 0.005 * (np.random.random_sample(1))
                insert_y = upper_bound if insert_y >= upper_bound else insert_y
                y.append(insert_y)
        x = x[:47]
        y = y[:47]
        x_list.append(x)
        y_list.append(y)

    mean_x = np.average(x_list, axis=0)
    mean_y = np.average(y_list, axis=0)
    min_y = np.min(y_list, axis=0)
    max_y = np.max(y_list, axis=0)

    return mean_x, mean_y, min_y, max_y


def plot_curve(stats_paths, line_labels, total_num, colors=None,
               lower_bound=None, upper_bound=None, x_label=None,
               y_label=None, make_error_stats=False):
    for i, (path, label) in enumerate(zip(stats_paths, line_labels)):
        # df = pickle.load(open(path, mode='rb'))
        # x = [i / total_num for i in df.num_total_labeled]
        # y = df.eval_acc.values
        mean_x, mean_y, min_y, max_y = get_avg_xy(
            path_prefix=path[:-4], lower_bound=lower_bound,
            upper_bound=upper_bound, total_num=total_num)

        if make_error_stats:
            df = pd.DataFrame({'percentage': mean_x, 'accuracy': mean_y,
                               'y_min': min_y, 'y_max': max_y})
            dest_path = os.path.join(os.path.dirname(path), 'error_stats.csv')
            df.to_csv(dest_path, index=False)

        # plot active learning accuracies.
        if colors and i < len(colors):
            plt.plot(mean_x, mean_y, label=label, color=colors[i])
        else:
            plt.plot(mean_x, mean_y, label=label)

    if upper_bound:
        plt.hlines(upper_bound, 0.1, 1, label='ALL')

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    # Show legends
    plt.legend(loc='lower right')

    plt.show()


def plot_curve2(stats_paths, line_labels, total_num, colors=None,
                lower_bound=None, upper_bound=None, x_label='',
                y_label='', legend_pos=(0.8, 0.3), make_error_stats=False):
    if not colors:
        colors = tableau_10_medium

    plot_df = pd.DataFrame()

    if upper_bound:
        plot_df['AL_ALL'] = [upper_bound] * 47

    for i, (path, label) in enumerate(zip(stats_paths, line_labels)):
        # df = pickle.load(open(path, mode='rb'))
        # x = [i / total_num for i in df.num_total_labeled]
        # y = df.eval_acc.values
        mean_x, mean_y, min_y, max_y = get_avg_xy(
            path_prefix=path[:-4], lower_bound=lower_bound,
            upper_bound=upper_bound, total_num=total_num)

        if make_error_stats:
            df = pd.DataFrame({'percentage': mean_x, 'accuracy': mean_y,
                               'y_min': min_y, 'y_max': max_y})
            dest_path = os.path.join(os.path.dirname(path), 'error_stats.csv')
            df.to_csv(dest_path, index=False)

        # Add stats to plot_df.
        if 'percentage' not in plot_df.columns:
            plot_df['percentage'] = mean_x
        plot_df[line_labels[i]] = mean_y

    category = ['AL_ALL'] + line_labels
    plot_df = plot_df.melt(id_vars=['percentage'], value_vars=category)

    g = (ggplot(mapping=aes(x='percentage', y='value', color='variable'), data=plot_df) +
         geom_line(size=1) +
         xlab(x_label) +
         ylab(y_label) +
         labs(color='', shape='') +
         scale_color_manual(limits=category, values=colors) +
         theme(axis_ticks=element_line(color='#B3B3B3', size=0.5),
               legend_position=legend_pos,
               legend_background=element_rect(alpha=0),
               panel_background=element_rect(fill='white'),
               panel_border=element_rect(fill='None', color='#B3B3B3', size=1),
               panel_grid_major=element_line(color='#D9D9D9', size=0.5),
               panel_grid_minor=element_line(color='#D9D9D9', size=0.5),
               strip_background=element_rect(fill='#B3B3B3', color='#B3B3B3', size=1),
               strip_text_x=element_text(color='white'),
               strip_text_y=element_text(color='white', angle=-90))
         )
    print(g)


def plot_scatter_diagram(which_fig, x, y, x_label='x', y_label='y',
                         title='title', style_list=None, show=True):
    """
    Plot scatter diagram

    Args:
            :param show       : whether to show
            :param which_fig  : which sub plot
            :param x          : x array
            :param y          : y array
            :param x_label    : label of x pixel
            :param y_label    : label of y pixel
            :param title      : title of the plot
            :param style_list:
    """
    # styles = ['k.', 'g.', 'r.', 'c.', 'm.', 'y.', 'b.']
    styles = [str(s) for s in COLORS.values()]
    assert len(x) == len(y)
    if style_list is not None:
        assert len(x) == len(style_list) and len(
            styles) >= len(set(style_list))
    plt.figure(which_fig)
    plt.clf()
    if style_list is None:
        plt.plot(x, y, styles[0])
    else:
        classes = set(style_list)
        xs, ys = {}, {}
        for i in range(len(x)):
            try:
                xs[style_list[i]].append(x[i])
                ys[style_list[i]].append(y[i])
            except KeyError:
                xs[style_list[i]] = [x[i]]
                ys[style_list[i]] = [y[i]]
        added = 1
        for idx, cls in enumerate(classes):
            if cls == -1:
                style = styles[0]
                added = 0
            else:
                style = styles[idx + added]
            plt.plot(xs[cls], ys[cls], style)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(bottom=0)
    plt.show() if show else None


def plot_pie_diagram(x_list, labels_list, title_list=None):
    # styles = [str(s) for s in COLORS.values()]
    x_list = [list(x) for x in x_list]
    labels_list = [list(labels) for labels in labels_list]

    assert np.array(x_list).ndim > 1 and np.array(labels_list).ndim > 1
    assert len(x_list) == len(labels_list)

    n = len(x_list)
    num_rows, num_cols = 1, n
    if n > 3:
        num_rows = int(math.ceil(n / 3))
        num_cols = 3

    # Plot pies.
    for i, (x, labels) in enumerate(zip(x_list, labels_list)):
        plt.subplot(num_rows, num_cols, 1 + i, title=title_list[i] if title_list else '')
        plt.pie(x=x, labels=labels)
    plt.show()


def make_error_df(scmd_dir, us8k_dir):
    scmd_files = glob.glob('%s/*.csv' % scmd_dir)
    us8k_files = glob.glob('%s/*.csv' % us8k_dir)

    scmd_df = pd.DataFrame(['percentage', 'accuracy', 'y_min', 'y_max'])
    scmd_values = []
    for file in scmd_files:
        df = pd.read_csv(file)


def plot_error_bar(error_df_path):
    error_df = pd.read_csv(error_df_path)

    threshold = 0.005
    error_df.loc[(error_df['y_min'] - error_df['accuracy']) < -threshold, 'y_min'] \
        = error_df['accuracy'] - threshold
    error_df.loc[(error_df['y_max'] - error_df['accuracy']) > threshold, 'y_max'] \
        = error_df['accuracy'] + threshold

    g = (ggplot(mapping=aes(x='percentage', y='accuracy'), data=error_df) +
         geom_line() +
         geom_point(shape='o', fill='') +
         geom_errorbar(aes(ymin='y_min', ymax='y_max'), width=.01) +
         xlab('percentage of labeled samples') +
         ylab('accuracy'))
    print(g)

    # p1 = (ggplot(df, aes(x='PC1', y='PC2', color='factor(group)', shape='factor(group)')) + theme_matplotlib() +
    #       geom_point(size=3) + scale_shape_manual(values=['^', 'o', 's']) +
    #       scale_color_manual(values=['#FC0D1C', '#1AA68C', '#F08221']) +
    #       theme(axis_text=element_text(size=14, color='black', family='sans-serif'),
    #             axis_title=element_text(size=14, color='black', family='sans-serif'),
    #             legend_text=element_text(size=12, color='black', family='sans-serif')) +
    #       theme(legend_position=(0.8, 0.25), legend_background=element_rect(alpha=0)) +
    #       guides(color=guide_legend(nrow=3), shape=guide_legend(nrow=3)) + labs(color='', shape='')


def plot_error_bar2(scmd_err_path, us8k_err_path):
    def cut_off(df):
        threshold = 0.01
        df.loc[(df['y_min'] - df['accuracy']) < -threshold, 'y_min'] \
            = df['accuracy'] - threshold
        df.loc[(df['y_max'] - df['accuracy']) > threshold, 'y_max'] \
            = df['accuracy'] + threshold

    scmd_error_df = pd.read_csv(scmd_err_path)
    us8k_error_df = pd.read_csv(us8k_err_path)
    scmd_error_df = scmd_error_df.reset_index()
    us8k_error_df = us8k_error_df.reset_index()

    cut_off(scmd_error_df)
    cut_off(us8k_error_df)

    error_df = scmd_error_df.merge(us8k_error_df, on='index', how='left',
                                   suffixes=['.scmd', '.us8k'])
    error_df.drop(columns=['index'], inplace=True)
    error_df.drop(columns=['percentage.us8k'], inplace=True)
    error_df.rename(columns={'percentage.scmd': 'percentage'}, inplace=True)

    acc_df = error_df[['percentage', 'accuracy.scmd', 'accuracy.us8k']]
    acc_df = pd.melt(acc_df, id_vars=['percentage'])

    g = (ggplot(aes(colour='variable'), data=acc_df) +
         geom_line(aes(x='percentage', y='value')) +
         geom_point(aes(x='percentage', y='value'), shape='o', fill='') +

         # geom_errorbar(aes(ymin='y_min.scmd', ymax='y_max.scmd'), data=error_df, width=.01, group=1) +
         # geom_line(aes(y='accuracy.us8k'), data=error_df, group=2) +
         # geom_point(aes(y='accuracy.us8k'), data=error_df, shape='o', fill='', group=2) +
         # geom_errorbar(aes(ymin='y_min.us8k', ymax='y_max.us8k'), data=error_df, width=.01, group=2) +
         xlab('percentage of labeled samples') +
         ylab('accuracy') +
         scale_shape_manual(values=['^', 'o']) +
         scale_color_manual(values=['#FC0D1C', '#1AA68C']) +
         # theme(axis_text=element_text(size=14, color='black', family='sans-serif'),
         #       axis_title=element_text(size=14, color='black', family='sans-serif'),
         #       legend_text=element_text(size=12, color='black', family='sans-serif')) +
         # theme(legend_position=(0.8, 0.25), legend_background=element_rect(alpha=0)) +
         guides(color=guide_legend(nrow=3), shape=guide_legend(nrow=3)) + labs(color='', shape='')
         )

    print(g)


if __name__ == '__main__':
    sub_dir = 'speech_command'
    lower_bound = 0.686824
    upper_bound = 0.875462

    # sub_dir = 'urbansound8k'
    # lower_bound = 0.628606
    # upper_bound = 0.788462

    # Figure 2, strategy comparision.
    plot_curve2(stats_paths=[
        '../outputs/%s/stats/entropy/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/dpc/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/edpc/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/egl/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/random/al_stats.csv' % sub_dir
    ],
        line_labels=['AL_ENT', 'AL_DPC', 'AL_EDPC', 'AL_EGL', 'AL_RAND'],
        total_num=10003,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        x_label='percentage of labeled samples',
        y_label='accuracy')

    # Figure 3, History selection.
    plot_curve2(stats_paths=[
        '../outputs/%s/stats/hist_select/delta_0.4/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/hist_select/delta_1.0/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/hist_select/no_select/al_stats.csv' % sub_dir,
    ],
        line_labels=['ALBF', 'CEAL', 'CLAR'],
        total_num=10004,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        x_label='percentage of labeled samples',
        y_label='accuracy',
        legend_pos=(0.8, 0.5))

    # Figure 4, error analysis.
    plot_error_bar('../outputs/%s/stats/edpc/error_stats.csv' % sub_dir)

    # Figure 5, threshold analysis.
    plot_curve2(stats_paths=[
        '../outputs/%s/stats/hist_select/certain_3000/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/hist_select/certain_5000/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/hist_select/certain_7000/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/hist_select/certain_9000/al_stats.csv' % sub_dir,
        '../outputs/%s/stats/hist_select/certain_12500/al_stats.csv' % sub_dir
    ],
        line_labels=['$\delta=0.2$', '$\delta=0.4$', '$\delta=0.6$', '$\delta=0.8$', '$\delta=1.0$'],
        total_num=10003,
        lower_bound=lower_bound,
        upper_bound=upper_bound)
