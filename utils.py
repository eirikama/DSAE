import matplotlib.pyplot as plt

def set_plotting_context(fig=None, 
                         ax=None, 
                         xlabel=r"$\tilde{\nu}\;\; (\frac{1}{cm})$",
                         ylabel='A', 
                         rotation_y=0, xlim=None, ylim=None):
    plt.style.use("ggplot")
    if fig is None:
        fig = plt.figure(figsize=(10, 6))
    if ax is None:
        ax = plt.gca()
        
    fig.frameon = True
    fig.edgecolor = 'red'
    ax.set_facecolor('gainsboro')    
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.invert_xaxis()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel(xlabel, fontsize=34)
    ax.set_ylabel(ylabel, rotation=rotation_y, fontsize=34, labelpad=18)    