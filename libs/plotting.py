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
            assignment=np.array([]), metric='correlation', row_cl=False):

    data = data_in.round()
    data_raw = data_raw_in.copy()

    height = int(data.shape[0] // 5)
    width = int(data.shape[1] // 7.5)
    fig, ax = plt.subplots(figsize=(width, height))

    if len(assignment) > 0:
        col_order = _get_col_order(assignment)

        clusters, cl_idx = np.unique(assignment, return_index=True)
        if clusters.size > len(COLORS):
            colors = get_colors(clusters.size)
            col_map = {i: next(colors) for i in clusters[np.argsort(cl_idx)]}
        else:
            col_map = {
                j: COLORS[i] for i,j in enumerate(clusters[np.argsort(cl_idx)])
            }

        col_dict = np.full(data_in.shape[1], '#ffffff', dtype='<U7')
        for i, cl in enumerate(data_in.columns[col_order]):
            try:
                col_dict[i] = col_map[cl]
            except:
                import pdb; pdb.set_trace()
        cluster_cols = pd.Series(col_dict, name='clusters', index=col_order)

        data.columns = np.arange(data_in.shape[1])
        data = data[col_order]

        if not data_raw.empty:
            data_raw.columns = np.arange(data_raw_in.shape[1])
            data_raw = data_raw[col_order]

            x_labels = data_raw_in.columns[col_order]
        else:
            x_labels = data_in.columns[col_order]
    else:
        x_labels = data_in.columns

    if row_cl:
        if not data_raw.empty:
            Z = linkage(data_raw.fillna(3), 'ward')
            row_order = dendrogram(Z, truncate_mode=None)['leaves']
        else:
            Z = linkage(data.fillna(3), 'ward')
            row_order = dendrogram(Z, truncate_mode=None)['leaves']

        data = data.iloc[row_order]
        if not data_raw.empty:
            data_raw = data_raw.iloc[row_order]
    else:
        row_order = np.arange(data.shape[0])

    if not data_raw.empty:
        annot = pd.DataFrame(
            np.full(data_raw.shape, '', dtype=str),
            index=data.index, columns=data.columns
        )
        annot[(data == 0) & (data_raw == 1)] = 'o'
        annot[(data == 1) & (data_raw == 0)] = 'x'
        annot[data_raw.isnull()] = '-'
    else:
        annot = False

    cmap = plt.get_cmap('Reds', 2)
    cmap.set_over('grey')

    cm = sns.clustermap(
        data, annot=annot, square=False, vmin=0, vmax=1, cmap=cmap, fmt='',
        linewidths=0, linecolor='lightgray', col_colors=cluster_cols,
        col_cluster=False, row_cluster=False #, col_colors_ratio=0.15
    )

    cm.cax.set_visible(False)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_heatmap.spines['top'].set_visible(True)
    cm.ax_heatmap.spines['right'].set_visible(True)
    cm.ax_heatmap.spines['bottom'].set_visible(True)
    cm.ax_heatmap.spines['left'].set_visible(True)

    cm.ax_heatmap.set_yticks(np.arange(0.5, data.shape[0], 1))
    cm.ax_heatmap.set_xticks(np.arange(0.5, data.shape[1], 1))

    cm.ax_heatmap.set_xticklabels(x_labels, rotation=90, fontsize=8)
    cm.ax_heatmap.set_yticklabels(data_in.index, fontsize=8)

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


def plot_traces(results, out_file=None, burn_in=0):
    if 'FP' in results[0].keys():
        errors = True
    else:
        errors = False

    if errors:
        fig = plt.figure(figsize=(10, 16))
        gs = GridSpec(8, 1)
        ax = {
            0: fig.add_subplot(gs[2, 0]),
            1: fig.add_subplot(gs[3, 0]),
            2: fig.add_subplot(gs[4:6, 0]),
            3: fig.add_subplot(gs[6:, 0]),
            4: fig.add_subplot(gs[0, 0]),
            5: fig.add_subplot(gs[1, 0])
        }
    else:
        fig = plt.figure(figsize=(10, 12))
        gs = GridSpec(6, 1)
        ax = {
            0: fig.add_subplot(gs[0, 0]),
            1: fig.add_subplot(gs[1, 0]),
            2: fig.add_subplot(gs[2:4, 0]),
            3: fig.add_subplot(gs[4:, 0])
        }

    plt.tick_params(axis='x', labelbottom=False)

    for chain, chain_result in enumerate(results):
        try:
            color = COLORS[chain]
        except IndexError:
            try:
                color = next(colors)
            except NameError:
                missing_cols = len(results) - len(COLORS)
                colors = get_colors(missing_cols)
                color = next(colors)

        _add_chain_traces(chain_result, ax, color, errors)

    stdout_fig(fig, out_file)


def _add_chain_traces(data, ax, color, errors, alpha=0.7):
    cl_no = [np.sum(~np.isnan(np.unique(i))) for i in data['assignments']]

    burn_in = data['burn_in']
    cl_no_mean, cl_no_std = ut._get_posterior_avg(cl_no[burn_in:])
    a_mean, a_std = ut._get_posterior_avg(data['DP_alpha'][burn_in:])

    ax[0].plot(data['DP_alpha'], color, alpha=alpha)
    ax[0].set_ylabel('DPMM\nalpha', fontsize=LABEL_FONTSIZE)
    ax[0].axhline(a_mean, ls='--', c=color)

    ax[1].plot(cl_no, color, alpha=alpha)
    ax[1].axhline(cl_no_mean, ls='--', c=color)

    # ax1.set_ylim(np.min(cl_no) - 1, np.max(cl_no[10:]) + 1)
    ax[1].set_ylabel('Cluster\nnumber', fontsize=LABEL_FONTSIZE)

    if data['MAP'].shape[0] != data['MAP'].size:
        for i, MAP in enumerate(data['MAP']):
            ax[2].plot(MAP, COLORS[i+1], alpha=alpha)
            ax[3].plot(data['ML'][i], COLORS[i+1], alpha=alpha)
    else:
        ax[2].plot(data['MAP'], color, alpha=alpha)
        ax[3].plot(data['ML'], color, alpha=alpha)
    ax[2].set_ylabel('Log a posteriori', fontsize=LABEL_FONTSIZE)
    ax[3].set_xlabel('MCMC steps', fontsize=LABEL_FONTSIZE)
    ax[3].set_ylabel('Log likelihood', fontsize=LABEL_FONTSIZE)

    if burn_in > 0:
        for ax_i in [0, 1, 2]:
            ax[ax_i].axvline(burn_in, c=color)
            ax[ax_i].get_xaxis().set_ticks([])
        ax[3].axvline(burn_in, c=color)

    if errors:
        FP_mean, FP_std = ut._get_posterior_avg(data['FP'][burn_in:])
        FN_mean, FN_std = ut._get_posterior_avg(data['FN'][burn_in:])

        ax[4].plot(data['FN'].round(4), color, alpha=alpha)
        ax[4].set_ylabel('FN error', fontsize=LABEL_FONTSIZE)
        ax[4].axhline(FN_mean, ls='--', c=color)
        ax[5].plot(data['FP'].round(4), color, alpha=alpha)
        ax[5].set_ylabel('FP error', fontsize=LABEL_FONTSIZE)
        ax[5].axhline(FP_mean, ls='--', c=color)

        if burn_in > 0:
            for ax_err in [4, 5]:
                ax[ax_err].axvline(burn_in, c=color)
                ax[ax_err].get_xaxis().set_ticks([])


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


if __name__ == '__main__':
    print('Here be dragons...')
