import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import colorcet as cc

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def plot_boxplots(result_df, x='nconc', y='confsums', n=100, directory="FiguresMpads/BoxPlots",save=True):
    palette = sns.color_palette(cc.glasbey_dark, n_colors=25)
    if x=='nconc':
        ax = sns.boxplot(data=result_df[result_df.ndrones == n], x=x, y=y, hue="methods", fill=False, palette=palette)
        ax.set_xlabel("Equal Priority Queue Length [-]")
        if y=="confsums":
            title = f"Total Conflicts for Varied Queue Length ({n} Drones)"
        else:
            title = f"Makespans for Varied Queue Length ({n} Drones)"
        
    else:
        ax = sns.boxplot(data=result_df[result_df.nconc == n], x=x, y=y, hue="methods", fill=False, palette=palette)
        ax.set_xlabel("Number of Flights to Plan [-]")
        if y=="confsums":
            title = f"Total Conflicts for Varied Number of Drones (Queue Length {n} Drones)"
        else:
            title = f"Makespans for Varied Number of Drones (Queue Length {n} Drones)"

    ylabel = "Total Conflicts [-]" if y=="confsums" else "Makespan [s]"
    
    # Create a custom legend with fill color
    legend_labels = result_df["methods"].unique()
    # legend_labels = ['sa', 'pso', 'ga', 'fc', 'mip']
    legend_handles = [plt.Line2D([0], [0], marker='o', color='white', 
                                #  markerfacecolor=sns.color_palette()[i], # trying to plot everything at once here
                                 markerfacecolor=palette[i],
                                 markersize=10, label=label) for i, label in enumerate(legend_labels)]
    # Add the custom legend to the plot
    # ax.legend(handles=legend_handles, title="Methods", loc='upper left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=False, shadow=False, title="Methods", handles=legend_handles)
    # ax.set_title(f"{title}")
    ax.set_ylabel(f"{ylabel}")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.grid(True, which='minor', color='black', lw=1)
    # reorderLegend(ax,['sa', 'pso', 'ga', 'fc', 'mip'])
    # ax.set_xticks([5., 10., 25.])
    # ax.set_xticklabels([5., 10., 25.])
    if save:
        plt.savefig(f"{directory}/{title}")
    plt.clf()
    return

# helper stuff
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels)
    return(handles, labels)

def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]


