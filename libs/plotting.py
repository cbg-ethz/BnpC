#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.cm import ScalarMappable

# Adjust matplotlib backend for snakemake
import psutil
if any(['snakemake' in i for i in psutil.Process().cmdline()]):
    import matplotlib
    matplotlib.use('Agg')

try:
    import libs.utils as ut
except ModuleNotFoundError:
    import utils as ut


COLORS = [
    '#1F78B4', '#33A02C', '#E31A1C', '#FF7F00', '#6A3D9A', # dark
    '#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F', '#CAB2D6', #light
    '#62A3CB', '#72BF5B', '#EF5A5A', '#FE9F37', '#9A77B8', # medium
    '#FFFF99', '#B15928', #ugly
]
TICK_FONTSIZE = 12
LABEL_FONTSIZE = 16


def get_colors(n, cmap='gist_rainbow', scale=0.85, alternating=True):
    def scale_color(col, scale):
        col_scaled = np.clip(col * scale, 0, 255).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*col_scaled)

    cm = plt.get_cmap(cmap)
    colors = np.apply_along_axis(
        scale_color, axis=1, arr=cm(np.arange(0, 1, 1 / n))[:, :-1] * 255,
        scale=scale
    )
    if alternating:
        colors1, colors2 = np.array_split(colors, 2)
        colors = np.full(n, '#000000', dtype='U7')
        np.place(colors, np.arange(n) % 2 == 0, colors1)
        np.place(colors, np.arange(n) % 2 == 1, colors2)
    return cycle(colors)


def _get_col_order(assignment):
    clusters = []
    cluster_idx = []
    doublets = np.zeros(len(assignment), dtype=int)

    for i, cl in enumerate(assignment):
        if isinstance(cl, (list, tuple, np.ndarray)):
            doublets[i] = 1
        elif cl not in clusters:
            clusters.append(cl)
            cluster_idx.append(i)

    col_order = np.array([], dtype=int)
    for cl_idx in np.argsort(cluster_idx):
        cols = [i for i, j in enumerate(assignment) \
            if isinstance(j, (int, np.integer)) and j == clusters[cl_idx]]
        col_order = np.append(col_order, cols)

    # Append doublet cells as last columns
    col_order = np.append(col_order, np.where(doublets == 1))
    return col_order


def plot_raw_data(data_in, data_raw_in=pd.DataFrame(), out_file=None,
            attachments=np.array([]), metric='correlation', x_labels=[],
            row_cl=False):

    data = data_in.copy()
    data_raw = data_raw_in.copy()

    height = int(data.shape[0] // 5)
    width = int(data.shape[1] // 7.5)
    fig, ax = plt.subplots(figsize=(width, height))

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    data.replace(3, np.nan, inplace=True)
    data.columns = np.arange(data.shape[1])

    if not isinstance(data_raw, pd.DataFrame):
        data_raw = pd.DataFrame(data_raw)

    if (data.size == data_raw.size) and (data.shape != data_raw.shape):
        data_raw = data_raw.T

    if len(attachments) > 0:
        cell_labels = np.array([
            -1 if isinstance(i, (list, tuple)) else int(i) for i in attachments
        ])
        dbts = (cell_labels < 0 ).astype(int)
        col_order = _get_col_order(cell_labels)
    else:
        col_order = data.columns
        cell_labels = data.columns
        dbts = np.zeros(len(cell_labels))

    clusters, cl_idx = np.unique(
        cell_labels[np.where(dbts == 0)], return_index=True
    )

    colors = get_colors(clusters.size)
    if clusters.size > len(COLORS):
        color_dict = {i:next(colors) for i in clusters[np.argsort(cl_idx)]}
    else:
        color_dict = {
            j:COLORS[i] for i,j in enumerate(clusters[np.argsort(cl_idx)])
        }

    cluster_cols = np.full(cell_labels.size, '#ffffff', dtype='<U7')

    for i, cl in enumerate(cell_labels[col_order]):
        try:
            cluster_cols[i] = color_dict[cl]
        except KeyError:
            pass
        except TypeError:
            break

    cluster_cols_s = pd.Series(cluster_cols, name='clusters')
    data = data[col_order]
    data.columns = np.arange(data.shape[1])

    if row_cl:
        if not data_raw.empty:
            Z = linkage(data_raw.fillna(3), 'ward')
            row_order = dendrogram(Z, truncate_mode=None)['leaves']
        else:
            Z = linkage(data.fillna(3), 'ward')
            row_order = dendrogram(Z, truncate_mode=None)['leaves']

        data = data.iloc[row_order]
    else:
        row_order = np.arange(data.shape[0])

    y_labels = data.index[row_order].tolist()

    if not data_raw.empty:
        data_raw = data_raw[col_order]
        data_raw.columns = np.arange(data.shape[1])
        data_raw = data_raw.iloc[row_order]

        annot = pd.DataFrame(np.full(data.shape, '', dtype=str))
        annot[(data == 0) & (data_raw == 1)] = 'o'
        annot[(data == 1) & (data_raw == 0)] = 'x'
        annot[data.isnull()] = '-'
    else:
        annot = False

    cmap = plt.get_cmap('Reds', 2)
    cmap.set_over('grey')

    cm = sns.clustermap(
        data.fillna(3), annot=annot, square=False, vmin=0, vmax=1,
        cmap=cmap, fmt='', linewidths=0, linecolor='lightgray',
        col_colors=cluster_cols_s, col_cluster=False, # col_colors_ratio=0.15
        row_cluster=False
    )

    cm.cax.set_visible(False)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_heatmap.spines['top'].set_visible(True)
    cm.ax_heatmap.spines['right'].set_visible(True)
    cm.ax_heatmap.spines['bottom'].set_visible(True)
    cm.ax_heatmap.spines['left'].set_visible(True)

    cm.ax_heatmap.set_yticks(np.arange(0.5, data.shape[0], 1))
    cm.ax_heatmap.set_xticks(np.arange(0.5, data.shape[1], 1))

    if any(x_labels):
        x_labels_final = x_labels[col_order]
    else:
        x_labels_final = np.arange(data.shape[1])[col_order]

    cm.ax_heatmap.set_xticklabels(x_labels_final, rotation=90, fontsize=8)
    cm.ax_heatmap.set_yticklabels(y_labels, fontsize=8)

    cm.gs.set_width_ratios([0, 0, 1])
    cm.gs.set_height_ratios([0, 0, 0.05, 0.95])
    cm.gs.update(left=0, bottom=0.00, right=1, top=1)

    if not out_file:
        plt.show()
    elif data.shape[0] < 50:
        cm.savefig(out_file, dpi=300)
    elif data.shape[0] < 100:
        cm.savefig(out_file, dpi=200)
    else:
        cm.savefig(out_file, dpi=100)
    plt.close()


def plot_cluster_data(data, out_file=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        data = data.append(data.iloc[1:].mean(), ignore_index=True)
        data = data.append(pd.Series(np.nan), ignore_index=True)
        old_idx = data.index.tolist()
        data = data.reindex([0] + old_idx[-2:] + old_idx[1:-2])
        data.rename(
            {0: 'params', old_idx[-2]: 'mean', old_idx[-1]: ''},
            axis=0, inplace=True
        )

    _add_cell_profile_to_ax(ax, data)
    stdout_fig(fig, out_file)


def plot_LL(results, out_file=None, burn_in=0):
    if 'ad_error' in results:
        errors = True
    else:
        errors = False

    if 'Z' in results:
        doublets = True
    else:
        doublets = False

    if errors and doublets:
        fig = plt.figure(figsize=(10, 16))
        gs = GridSpec(11, 1)

        ax0 = fig.add_subplot(gs[5, 0])
        ax1 = fig.add_subplot(gs[6, 0])
        ax2 = fig.add_subplot(gs[7:9, 0])
        ax3 = fig.add_subplot(gs[9:, 0])

        ax4 = fig.add_subplot(gs[0, 0])
        ax5 = fig.add_subplot(gs[1, 0])

        ax6 = fig.add_subplot(gs[2, 0])
        ax7 = fig.add_subplot(gs[3, 0])
        ax8 = fig.add_subplot(gs[4, 0])
    elif errors and not doublets:
        fig = plt.figure(figsize=(10, 16))
        gs = GridSpec(8, 1)

        ax0 = fig.add_subplot(gs[2, 0])
        ax1 = fig.add_subplot(gs[3, 0])
        ax2 = fig.add_subplot(gs[4:6, 0])
        ax3 = fig.add_subplot(gs[6:, 0])

        ax4 = fig.add_subplot(gs[0, 0])
        ax5 = fig.add_subplot(gs[1, 0])
    elif not errors and doublets:
        fig = plt.figure(figsize=(10, 12))
        gs = GridSpec(9, 1)

        ax0 = fig.add_subplot(gs[3, 0])
        ax1 = fig.add_subplot(gs[4, 0])
        ax2 = fig.add_subplot(gs[5:7, 0])
        ax3 = fig.add_subplot(gs[7:, 0])

        ax6 = fig.add_subplot(gs[0, 0])
        ax7 = fig.add_subplot(gs[1, 0])
        ax8 = fig.add_subplot(gs[2, 0])
    else:
        fig = plt.figure(figsize=(10, 12))
        gs = GridSpec(6, 1)

        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2:4, 0])
        ax3 = fig.add_subplot(gs[4:, 0])

    plt.tick_params(axis='x', labelbottom=False)

    if 'Z' in results:
        cl_no = (results['Z'].sum(axis=1) > 0).sum(axis=1)
    else:
        cl_no = [np.sum(~np.isnan(np.unique(i))) for i in results['assignments']]

    if burn_in:
        cl_no_mean, cl_no_std = ut._get_posterior_avg(cl_no[burn_in:])
        a_mean, a_std = ut._get_posterior_avg(results['DP_alpha'][burn_in:])
    else:
        cl_no_mean, cl_no_std = ut._get_posterior_avg(cl_no)
        a_mean, a_std = ut._get_posterior_avg(results['DP_alpha'])

    ax0.plot(results['DP_alpha'], 'darkolivegreen')
    ax0.set_ylabel('DPMM\nalpha', fontsize=LABEL_FONTSIZE)
    ax0.axhline(a_mean, ls='--')
    ax0.axhline(a_mean - a_std, ls=':')
    ax0.axhline(a_mean + a_std, ls=':')

    ax1.plot(cl_no, 'purple')
    ax1.axhline(cl_no_mean, ls='--')
    ax1.axhline(cl_no_mean - cl_no_std, ls=':')
    ax1.axhline(cl_no_mean + cl_no_std, ls=':')

    ax1.set_ylim(np.min(cl_no) - 1, np.max(cl_no[10:]) + 1)
    ax1.set_ylabel('Cluster\nnumber', fontsize=LABEL_FONTSIZE)

    ax2.plot(results['MAP'], 'black')
    ax2.set_ylabel('Log a posteriori', fontsize=LABEL_FONTSIZE)

    ax3.plot(results['ML'], 'black')
    ax3.set_xlabel('MCMC steps', fontsize=LABEL_FONTSIZE)
    ax3.set_ylabel('Log likelihood', fontsize=LABEL_FONTSIZE)

    if errors:
        if burn_in:
            FP_mean, FP_std = ut._get_posterior_avg(results['fd_error'][burn_in:])
            FN_mean, FN_std = ut._get_posterior_avg(results['ad_error'][burn_in:])
        else:
            FP_mean, FP_std = ut._get_posterior_avg(results['fd_error'])
            FN_mean, FN_std = ut._get_posterior_avg(results['ad_error'])
        ax4.plot(results['ad_error'].round(4), 'blue')
        ax4.set_ylabel('FN error', fontsize=LABEL_FONTSIZE)
        ax4.axhline(FN_mean, ls='--')
        ax4.axhline(FN_mean - FN_std, ls=':')
        ax4.axhline(FN_mean + FN_std, ls=':')
        ax5.plot(results['fd_error'].round(4), 'green')
        ax5.set_ylabel('FP error', fontsize=LABEL_FONTSIZE)
        ax5.axhline(FP_mean, ls='--')
        ax5.axhline(FP_mean - FP_std, ls=':')
        ax5.axhline(FP_mean + FP_std, ls=':')
    if doublets:
        dbt_no = (results['Z'].sum(axis=2) - 1).sum(axis=1)
        ax6.plot(dbt_no, 'green')
        ax6.set_ylabel('Doublet\nnumber', fontsize=LABEL_FONTSIZE)
        ax6.set_ylim(0, np.max(dbt_no) + 1)

        ax7.plot(results['d'], 'green')
        ax7.set_ylabel('Doublet\nrate', fontsize=LABEL_FONTSIZE)
        ax7.set_ylim(0, np.max(results['d']) + 0.1)

        cl_w_norm = results['pi'].T / results['pi'].T.sum(axis=0)
        for i, cl_w in enumerate(cl_w_norm):
            ax8.plot(cl_w, COLORS[i])
        ax8.set_ylabel('Class\nweights', fontsize=LABEL_FONTSIZE)
        ax8.set_ylim(-.05, 1.05)

    if burn_in:
        for ax in [ax0, ax1, ax2]:
            ax.axvline(burn_in, c='r')
            ax.get_xaxis().set_ticks([])
        ax3.axvline(burn_in, c='r')

        if errors:
            for ax_err in [ax4, ax5]:
                ax_err.axvline(burn_in, c='r')
                ax_err.get_xaxis().set_ticks([])
        if doublets:
            for ax_dbt in [ax6, ax7, ax8]:
                ax_dbt.axvline(burn_in, c='r')
                ax_dbt.get_xaxis().set_ticks([])

    stdout_fig(fig, out_file)


def plot_clusters(arr_obs, df_pred_in, assign_pred, names, out_file):
    if arr_obs.shape == df_pred_in.shape:
        df_obs = pd.DataFrame(arr_obs, index=names[0])
    else:
        df_obs = pd.DataFrame(arr_obs.T, index=names[1])

    if df_pred_in.shape[1] != len(assign_pred):
        df_pred = ut._get_genotype_all(df_pred_in, assign_pred)
    else:
        df_pred = df_pred_in.copy()

    try:
        df_pred.index = names[0]
        df_pred.columns = names[1]
    except ValueError:
        df_pred.index = names[1]
        df_pred.columns = names[0]

    plot_raw_data(
        df_pred, df_obs, attachments=assign_pred, out_file=out_file,
        x_labels=df_pred.columns
    )


def plot_doublets(data, out_file=None, thresh=0.5):
    fig, ax = plt.subplots(figsize=(10, 5))

    if data.ndim == 1:
        x_ticks = np.arange(data.size)
        plt.plot(np.arange(data.size), data, color='r', ms=12,
            marker='x', linestyle=''
        )
        dbt_idx = np.array([])
    else:
        x_ticks = np.arange(data.shape[1])
        cell_assign = data.sum(axis=2)
        plt.errorbar(
            x=x_ticks, y=cell_assign.mean(axis=0) - 1,
            yerr=cell_assign.std(axis=0),
            fmt='xk', mec='r', ms=10, capsize=5, lw=1
        )
        dbt_idx = np.where(
            (data.sum(axis=0).sum(axis=1) - data.shape[0]) / data.shape[0] > thresh
        )[0]

    x_names = ['{} *'.format(i) if i in dbt_idx else str(i) for i in x_ticks]
    plt.xticks(x_ticks, x_names, rotation=90, fontsize=LABEL_FONTSIZE*.5)

    ax.axhline(.5, c='#FF00009F', ls='--', lw='1')
    ax.axhline(0, c='k', ls='--', lw='1')
    ax.axhline(1, c='k', ls='--', lw='1')

    ax.set_xlabel('Cells', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Doublet Probability', fontsize=LABEL_FONTSIZE)

    stdout_fig(fig, out_file)



def plot_similarity(data, out_file=None, attachments=None):
    cmap='OrRd'

    fig, ax = plt.subplots(figsize=np.clip(np.array(data.shape) * 0.3, 1, 50))

    if attachments:
        col_order = _get_col_order(attachments)

        data = pd.DataFrame(data)
        data = data[col_order]

        data = data.reindex(col_order)

    hm = sns.heatmap(
        data,  ax=ax, annot_kws={'size': 6}, annot=True, fmt='.2f',
        linewidths=.5, square=True, linecolor='lightgray',
        cmap=cmap, cbar_kws={'shrink': .5}, vmin=0, vmax=1,
    )

    ax.set_ylabel('Cell', fontsize=LABEL_FONTSIZE)
    ax.set_xlabel('Cell', fontsize=LABEL_FONTSIZE)
    ax.set_title('Pairwise Similarity Matrix', fontsize=LABEL_FONTSIZE)

    if data.shape[0] < 50:
        stdout_fig(fig, out_file)
    elif data.shape[0] < 100:
        stdout_fig(fig, out_file, dpi=200)
    else:
        stdout_fig(fig, out_file, dpi=100)


def plot_cluster_number(data, out_file=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.plot(data)

    ax.set_xlabel('MCMC steps', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Number of clusters', fontsize=LABEL_FONTSIZE)

    stdout_fig(fig, out_file)


def plot_error_LL(x, y, curr_err, error_type='beta', out_file=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    plt.plot(x, y)

    ax.axvline(curr_err, c='r')
    ax.set_xlabel(r'$\{}$ Error'.format(error_type), fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('log Likelihood', fontsize=LABEL_FONTSIZE)

    stdout_fig(fig, out_file)


def color_tree_nodes(tree_file, clusters, out_dir='', transpose=True,
            prefix='colored'):
    with open(tree_file, 'r') as f_in:
        gv_raw = f_in.read().rstrip('}')

    if len(re.findall('circle', gv_raw)) > 1:
        circle_pos = gv_raw.rfind('circle')
        gv_raw = gv_raw[:circle_pos] + 'square' + gv_raw[circle_pos+6:]

    clusters = [-1 if isinstance(i, tuple) else i for i in clusters]

    colors = get_colors(np.unique(clusters).size)

    cluster_cols = {i: next(colors) for i in np.unique(clusters)}
    # White for doublet cells
    cluster_cols[-1] = '#ffffff'

    if transpose:
        for cell, cluster in enumerate(clusters):
            gv_raw += 's{:02d} [fillcolor="{}"];\n' \
                .format(cell, cluster_cols[cluster])
    else:
        for mut, cluster in enumerate(clusters):
            gv_raw += '{} [fillcolor="{}"];\n' \
                .format(mut+1, cluster_cols[cluster])
    gv_raw += '}'

    out_file = os.path.join(
        out_dir,
        os.path.basename(tree_file).replace('.gv', '__{}.gv'.format(prefix))
    )

    with open(out_file, 'w') as f_out:
        f_out.write(gv_raw)

    try:
        from graphviz import render
        render('dot', 'png', out_file)
    except ImportError:
        pass


def plot_cluster_probs(x, y, heatmap_data, out_file=None):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(3, 1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1:, 0])

    _add_cell_profile_to_ax(ax0, heatmap_data)
    _add_distribution_to_ax(ax1, x, y)

    stdout_fig(fig, out_file)


def stdout_fig(fig, out_file, dpi=300):
    if not out_file:
        try:
            fig.tight_layout()
        except AttributeError:
            pass
        plt.show()
    else:
        try:
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
        except AttributeError:
            pass
        fig.savefig(out_file, dpi=dpi)
        plt.close()


def _add_distribution_to_ax(ax, x, y):
    bars = ax.bar(x, y, width = 0.75)

    ax.set_ylim([0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90)
    ax.set_xlabel('Cluster number', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Probability', fontsize=LABEL_FONTSIZE)


def _add_cell_profile_to_ax(ax, data):
    sns.heatmap(
        data[data.columns[::-1]].T, annot=True, fmt='.2f', linewidths=.5, ax=ax,
        square=True, vmin=0, vmax=1, cmap='RdYlBu_r', annot_kws={'size': 6},
        linecolor='lightgray', cbar=False
    )


def plot_traces(traces, x, title='', out_file=''):
    colors = get_colors(traces.shape[0])
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, trace in enumerate(traces):
        plt.plot(
            x, trace, lw=1, alpha=0.2, color=next(colors),
            label='run {:02d}'.format(idx)
        )

    if traces.shape[0] > 2:
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0)

        plt.plot(x, mean_trace, lw=2, color='r', label='mean trace')

        tprs_upper = mean_trace + std_trace
        tprs_lower = mean_trace - std_trace
        plt.fill_between(
            x, tprs_lower, tprs_upper, color='grey', alpha=.75,
            label=r'$\pm$ 1 std. dev.'
        )

    if title:
        plt.title(title)
    plt.ylabel('log likelihood')
    plt.xlabel('# MCMC steps')
    plt.legend()

    stdout_fig(fig, out_file)


def plot_parameter_traces(params, assignment, burn_in, out_dir):
    params_dir = os.path.join(out_dir, 'parameter_traces')
    os.mkdir(params_dir)
    for cluster, data in enumerate(params):
        cells = np.sum(assignment == cluster, axis=1)
        if cells[burn_in:].sum() > 0:
            cl_out_file = os.path.join(
                params_dir, 'cluster_{:02d}.png'.format(cluster)
            )
            _plot_cluster_trace(data, cells, cl_out_file)


def _plot_cluster_trace(data, cells, out_file):
    fig = plt.figure(figsize=(10, data.shape[0]))
    gs = GridSpec(data.shape[0] + 1, 1)

    # plot cells per cluster
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(cells, 'r')
    ax.set_ylabel('# Cells', fontsize=LABEL_FONTSIZE / 2)
    ax.get_xaxis().set_visible(False)
    ax.set_xlim([0, data.shape[1]])

    # plot item traces
    for idx, trace in enumerate(data):
        ax = fig.add_subplot(gs[idx+1, 0])
        ax.plot(trace, 'k')
        ax.set_ylabel('Item {:02d}'.format(idx), fontsize=LABEL_FONTSIZE / 2)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, data.shape[1]])
        ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('MCMC steps', fontsize=LABEL_FONTSIZE)

    stdout_fig(fig, out_file)


if __name__ == '__main__':
    print('Here be dragons...')
