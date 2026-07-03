import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from utils import params_to_text, get_suffix

def draw_heatmap(matrix,x_range,y_range,x_param_name,y_param_name,folder_name='figs',global_max=None,global_min=None,figsize=(6,4.5),cmap='jet',cbar_label='',params=None,params_varied=[],string_varied=[],t_end=500,TREAT=False):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
 
    cmap = plt.get_cmap(cmap)

    f,ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix,cmap = cmap, vmin=global_min,vmax=global_max,origin='lower')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")

    cbar = f.colorbar(im,cax=cax)
    cbar.set_label(cbar_label)
    if global_max==1 and global_min==0:
        tick_locs = np.array([0,0.5,1])
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([str(el) for el in tick_locs])
    else:
        tick_locs = np.linspace(np.nanmin(matrix), np.nanmax(matrix), 3)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([str(np.round(el,2)) for el in tick_locs])
    cbar.ax.yaxis.set_tick_params(which='both',length=0)
        
    ax.tick_params(axis='both', which='major',length=0)

    n_yticks = 5
    indices = np.array(np.linspace(0,len(y_range)-1, n_yticks),dtype=int)
    ax.set_ylabel(y_param_name)
    round_precision_y = 3 if min(y_range)<= 0 else max(0,min(3,1-int(np.floor(np.log10(min(y_range))))))
    y_axis_ticks_vals = np.array([str(np.round(val,round_precision_y)) for val in  y_range[indices]])      
    ax.set_yticks(indices)
    ax.set_yticklabels(list(map(str,y_axis_ticks_vals)))

    n_xticks = 5
    indices = np.array(np.linspace(0,len(x_range)-1, n_xticks),dtype=int)
    ax.set_xlabel(x_param_name)
    round_precision_x = 3 if min(x_range)<= 0 else max(0,min(3,1-int(np.floor(np.log10(min(x_range))))))
    x_axis_ticks_vals = np.array([str(np.round(val,round_precision_x)) for val in  x_range[indices]]) 
    ax.set_xticks(indices)
    ax.set_xticklabels(list(map(str,x_axis_ticks_vals)))

    
    if not params is None:
        text_left, text_right = params_to_text(params, params_varied=[x_param_name, y_param_name] if params_varied == [] else params_varied, string_varied=['x', 'y'] if string_varied == [] else string_varied, TREAT=TREAT)
        [x1, x2] = ax.get_xlim()
        [y1, y2] = ax.get_ylim()
        ax.text(x1 + 0.1 * (x2 - x1), y2 + 0.1 * (y2 - y1), '\n'.join(text_left), va='bottom', ha='center')
        ax.text(x1 + 0.9 * (x2 - x1), y2 + 0.1 * (y2 - y1), '\n'.join(text_right), va='bottom', ha='center')
        suffix = get_suffix(text_left, text_right) + '_tend' + str(t_end)
        identifer_type = ''.join([el[0] for el in cbar_label.split(' ')])
        plt.savefig(os.path.join(folder_name, 'sens_2d_' + identifer_type + '_' + suffix + '.pdf'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder_name, 'sens_2d_' + cbar_label.replace(' ', '_') + '.pdf'), bbox_inches='tight')