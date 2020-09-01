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

# Adjust matplotlib backend for snakemake/cluster
try:
    plt.figure()
except:
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
    clusters, cluster_cnt = np.unique(assignment, return_counts=True)
   
    col_order = np.array([], dtype=int)
    for cl_idx in np.argsort(cluster_cnt)[::-1]:
        cols = [i for i, j in enumerate(assignment) \
            if j == clusters[cl_idx]]
        col_order = np.append(col_order, cols)

    return col_order


def plot_raw_data(data_in, data_raw_in=pd.DataFrame(), out_file=None,
            assignment=np.array([]), metric='correlation', row_cl=True):

    data = data_in.copy()
    data_raw = data_raw_in.copy()

    height = int(data.shape[0] // 5)
    width = int(data.shape[1] // 7.5)
    fig, ax = plt.subplots(figsize=(width, height))

    if len(assignment) > 0:
        col_order = _get_col_order(assignment)

        clusters, cl_cnt = np.unique(assignment, return_counts=True)

        if clusters.size > len(COLORS):
            colors = get_colors(clusters.size - len(COLORS))

        col_map =  {}
        for i, j in enumerate(clusters[np.argsort(cl_cnt)[::-1]]):
            try:
                col_map[j] = COLORS[i] 
            except IndexError:
                col_map[j] = next(colors)

        col_dict = np.full(data_in.shape[1], '#ffffff', dtype='<U7')
        for i, cl in enumerate(col_order):
            col_dict[i] = col_map[assignment[cl]]
                
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
        Z = linkage(data.fillna(3), 'complete')
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
        if data.min().min() < 0:
            annot[(data.round() == -1) & (data_raw == 1)] = 'o'
        else:
            annot[(data.round() == 0) & (data_raw == 1)] = 'o'
        annot[(data.round() == 1) & (data_raw == 0)] = 'x'
        annot[data_raw.isnull()] = '-'
    else:
        annot = False

    # cmap = plt.get_cmap('bwr', 100)
    # cmap = plt.get_cmap('Reds', 100)
    cmap = plt.get_cmap('Reds', 2)

    cmap.set_over('green')
    cmap.set_bad('grey')

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
    no_rows = 6

    if 'FP' in results[0].keys():
        no_rows += 2
        errors = True
    else:
        errors = False

    if 'PSRF' in results[0].keys():
        no_rows += 1
        psrf = True
    else:
        psrf = False

    fig = plt.figure(figsize=(10, no_rows * 2))
    gs = GridSpec(no_rows, 1)
    ax = {0: fig.add_subplot(gs[0, 0]),
        1: fig.add_subplot(gs[1, 0]),
        2: fig.add_subplot(gs[2:4, 0]),
        3: fig.add_subplot(gs[4:6, 0])}
    if errors:
        ax[4] = fig.add_subplot(gs[6, 0])
        ax[5] = fig.add_subplot(gs[7, 0])

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

        _add_chain_traces(chain_result, ax, color)

    step_no = chain_result['ML'].size + 1
    if psrf:
        ax[6] = fig.add_subplot(gs[no_rows - 1, 0])
        psrf_val = np.full(step_no, np.nan)
        for step_i, psrf_i in chain_result['PSRF']:
            psrf_val[step_i] = psrf_i
        ax[6].plot(np.arange(step_no), psrf_val, 'rx')
        ax[6].set_ylabel('PSRF', fontsize=LABEL_FONTSIZE)
        ax[6].axhline(1, ls='-', c='black')
        ax[6].axhline(chain_result['PSRF_cutoff'], ls=':', c='red')

    # Add x-axis label and tick labels below last plot, remove from others
    tick_dist = int(np.floor(step_no // 10 / 100) * 100)
    tick_pos = [tick_dist * i for i in range(0, 11, 1)]

    last_ax = max(ax.keys())
    for ax_id, ax_obj in ax.items(): 
        ax_obj.set_xlim(-step_no * 0.05, step_no * 1.05)
        ax_obj.set_xticks(tick_pos)
        if ax_id == last_ax:
            ax_obj.set_xticklabels([str(i) for i in tick_pos])
            ax_obj.set_xlabel('MCMC steps', fontsize=LABEL_FONTSIZE)
        else:
            ax_obj.set_xticklabels([])

    stdout_fig(fig, out_file)

    
def _add_chain_traces(data, ax, color, alpha=0.4, std_fkt=2.576):
    burn_in = data['burn_in']

    a_mean, a_std = ut._get_posterior_avg(data['DP_alpha'][burn_in:])
    ax[0].plot(data['DP_alpha'], color, alpha=alpha)
    ax[0].set_ylabel('DPMM\nalpha', fontsize=LABEL_FONTSIZE)
    ax[0].axhline(a_mean, ls='--', c=color)
    ax[0].set_ylim(a_mean - std_fkt * a_std, a_mean + std_fkt * a_std)

    cl = [np.sum(~np.isnan(np.unique(i))) for i in data['assignments']]
    cl_mean, cl_std = ut._get_posterior_avg(cl[burn_in:])
    ax[1].plot(cl, color, alpha=alpha)
    ax[1].axhline(cl_mean, ls='--', c=color)
    ax[1].set_ylim(cl_mean - std_fkt * cl_std, cl_mean + std_fkt * cl_std)
    ax[1].plot(cl, color, alpha=alpha)
    ax[1].axhline(cl_mean, ls='--', c=color)
    ax[1].set_ylabel('Cluster\nnumber', fontsize=LABEL_FONTSIZE)

    if data['MAP'].shape[0] != data['MAP'].size:
        for i, MAP in enumerate(data['MAP']):
            ax[2].plot(MAP, COLORS[i+1], alpha=alpha)
            ax[3].plot(data['ML'][i], COLORS[i+1], alpha=alpha)
    else:
        ax[2].plot(data['MAP'], color, alpha=alpha)
        ax[3].plot(data['ML'], color, alpha=alpha)
    ax[2].set_ylabel('Log a posteriori', fontsize=LABEL_FONTSIZE)
    ax[3].set_ylabel('Log likelihood', fontsize=LABEL_FONTSIZE)

    if 4 in ax:
        FN_mean, FN_std = ut._get_posterior_avg(data['FN'][burn_in:])
        ax[4].plot(data['FN'].round(4), color, alpha=alpha)
        # ax[4].set_ylim(FN_mean - std_fkt * FN_std, FN_mean + std_fkt * FN_std)
        ax[4].set_ylabel('FN error', fontsize=LABEL_FONTSIZE)
        ax[4].axhline(FN_mean, ls='--', c=color)
    if 5 in ax:
        FP_mean, FP_std = ut._get_posterior_avg(data['FP'][burn_in:])
        ax[5].plot(data['FP'].round(4), color, alpha=alpha)
        # ax[5].set_ylim(FP_mean - std_fkt * FP_std, FP_mean + std_fkt * FP_std)
        ax[5].set_ylabel('FP error', fontsize=LABEL_FONTSIZE)
        ax[5].axhline(FP_mean, ls='--', c=color)

    if burn_in > 0:
        for ax_id, ax_obj in ax.items():
            ax_obj.axvline(burn_in, c=color)


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
            gv_raw += f's{cell:02d} [fillcolor="{cluster_cols[cluster]}"];\n'
    else:
        for mut, cluster in enumerate(clusters):
            gv_raw += f'{mut+1} [fillcolor="{cluster_cols[cluster]}"];\n'
    gv_raw += '}'

    out_file = os.path.join(
        out_dir,
        os.path.basename(tree_file).replace('.gv', f'__{prefix}.gv')
    )

    with open(out_file, 'w') as f_out:
        f_out.write(gv_raw)

    try:
        from graphviz import render
        render('dot', 'png', out_file)
    except:
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


def load_txt(path):
    try:
        df = pd.read_csv(path, sep='\t', index_col=False)
        x = df.at[0, 'Assignment'].strip().split(' ')
    except ValueError:
        with open(path, 'r') as f:
            x = f.read().strip().split(' ')

    return [int(i) for i in x]


if __name__ == '__main__':
    print('Here be dragons...')

