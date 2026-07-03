"""
Created on Tue Jan  7 21:10:18 2025

@author: ckadelka
"""

# %%
import numpy as np
import os

import model
from utils import save_data, load_data, params_to_text
from draw_heatmap import draw_heatmap

# %%
def count_number_of_changes_0_to_1(binary_list):
    count = 0
    for i in range(len(binary_list) - 1):
        if binary_list[i] == 0 and binary_list[i + 1] == 1:
            count += 1
    return count

def sensitivity_2d(x_range, y_range, x_param_name, y_param_name, dt=1, t_end=500, dict_non_default_parameters={}):
    final_proportion_sensitive_cells = np.zeros((len(y_range), len(x_range)))
    mean_proportion_sensitive_cells = np.zeros((len(y_range), len(x_range)))
    n_treatment_turned_on = np.zeros((len(y_range), len(x_range)))
    proportion_treatment_on  = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            parameters = dict_non_default_parameters.copy()
            parameters.update({x_param_name: x, y_param_name: y, 'dt': dt, 't_end': t_end})
            ts, solution, model_params = model.simulate(**parameters)
            final_proportion_sensitive_cells[i, j] = solution[-1,0] / (solution[-1,0] + solution[-1,1])
            mean_proportion_sensitive_cells[i, j] = sum(solution[:,0])/solution.sum()
            # n_treatment_turned_on[i, j] = count_number_of_changes_0_to_1(treatment)
            # proportion_treatment_on[i, j] = np.mean(treatment)
    return (final_proportion_sensitive_cells, mean_proportion_sensitive_cells), model_params

def sensitivity_2d_adaptive_therapy(x_range, y_range, x_param_name, y_param_name, dt=1, t_end=500, dict_non_default_parameters={}):
    final_proportion_sensitive_cells = np.zeros((len(y_range), len(x_range)))
    mean_proportion_sensitive_cells = np.zeros((len(y_range), len(x_range)))
    n_treatment_turned_on = np.zeros((len(y_range), len(x_range)))
    proportion_treatment_on  = np.zeros((len(y_range), len(x_range)))
    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            parameters = dict_non_default_parameters.copy()
            parameters.update({x_param_name: x, y_param_name: y, 'dt': dt, 't_end': t_end})
            ts, solution, treatment, model_params = model.simulate_adaptive_therapy(**parameters)
            final_proportion_sensitive_cells[i, j] = solution[-1,0] / (solution[-1,0] + solution[-1,1])
            mean_proportion_sensitive_cells[i, j] = sum(solution[:,0])/solution.sum()
            n_treatment_turned_on[i, j] = count_number_of_changes_0_to_1(treatment)
            proportion_treatment_on[i, j] = np.mean(treatment)
    return (final_proportion_sensitive_cells, mean_proportion_sensitive_cells, n_treatment_turned_on, proportion_treatment_on), model_params


def plot_2d_sensitivity(x_param_name,x_min,x_max,y_param_name,y_min,y_max,n_steps=20,x_scale='linear',y_scale='linear',dt=1,t_end=500, dict_non_default_parameters={},figsize=(4,3),PLOT=True,TREAT=False):
    hashed = hash((x_param_name,x_min,x_max,x_scale,y_param_name,y_min,y_max,y_scale,n_steps,dt,t_end,tuple(dict_non_default_parameters.keys()),tuple(dict_non_default_parameters.values())))
    hashed_str = str(hashed).replace('-','m')
    try:
        data = load_data('sensitivity_matrices_%s.pkl' % (hashed_str))
    except FileNotFoundError:
        if x_scale=='linear':
            x_range = np.linspace(x_min,x_max,n_steps)
        else:
            x_range = np.logspace(np.log10(x_min),np.log10(x_max),n_steps)
        if y_scale=='linear':
            y_range = np.linspace(y_min,y_max,n_steps)
        else:
            y_range = np.logspace(np.log10(y_min),np.log10(y_max),n_steps)    

        if TREAT:
            matrices, params = sensitivity_2d_adaptive_therapy(x_range, y_range, x_param_name,y_param_name, dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)
        else:
            matrices,params = sensitivity_2d(x_range, y_range, x_param_name,y_param_name, dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)
        data = {
            'x_range': x_range,
            'y_range': y_range,
            'dict_non_default_parameters' : dict_non_default_parameters,
            'matrices': matrices,
            'params' : params
        }
        save_data(data,'sensitivity_matrices_%s.pkl' % (hashed_str))
    
    if PLOT:
        params_varied = [x_param_name,y_param_name]
        string_varied = ['x','y']
        for key in data['dict_non_default_parameters']:
            if data['dict_non_default_parameters'][key] == x_param_name:
                params_varied.append(key)
                string_varied.append('x')
            if data['dict_non_default_parameters'][key] == y_param_name:
                params_varied.append(key)
                string_varied.append('y')
                
        cbar_labels = ['final proportion sensitive cells','average proportion sensitive cells','number of treatments']
        cmaps = ['RdBu','RdBu','Greens']
        
        for ii,(cbar_label,cmap) in enumerate(zip(cbar_labels,cmaps)):
            draw_heatmap(data['matrices'][ii],data['x_range'],data['y_range'],x_param_name,y_param_name,'figs',figsize=figsize,cmap=cmap,cbar_label=cbar_label,params=data['params'],params_varied=params_varied,string_varied=string_varied,t_end=t_end,TREAT=TREAT)
        return
    else: #used for 4d sensitivity plots
        return data
    
def plot_4d_sensitivity(x_param_name,x_min,x_max,y_param_name,y_min,y_max,xx_param_name,xx1,xx2,yy_param_name,yy1,yy2,n_steps=20,x_scale='linear',y_scale='linear',dt=1,t_end=500, dict_non_default_parameters={},figsize=(8,6),folder_name='figs', TREAT=False):
    import matplotlib.pyplot as plt
    from utils import get_suffix
    
    if not os.path.exists(folder_name) and len(folder_name)>0:
        os.makedirs(folder_name)
    if TREAT==True:
        cbar_labels = ['final proportion sensitive cells','average proportion sensitive cells','number of treatments']
        cmaps = ['RdBu','RdBu','Greens']
    else:
        cbar_labels = ['final proportion sensitive cells','average proportion sensitive cells']
        cmaps = ['RdBu','RdBu']

    for ii,(cbar_label,cmap) in enumerate(zip(cbar_labels,cmaps)):
        data_4d = np.zeros((2,2,n_steps,n_steps))
        for i,xx_val in enumerate([xx1,xx2]):
            for j,yy_val in enumerate([yy1,yy2]):
                dict_parameters = dict_non_default_parameters.copy()
                dict_parameters.update({xx_param_name:xx_val,yy_param_name:yy_val})
                data_2d = plot_2d_sensitivity(x_param_name,x_min,x_max,y_param_name,y_min,y_max,n_steps=n_steps,x_scale=x_scale,y_scale=y_scale,dt=dt,t_end=t_end,dict_non_default_parameters=dict_parameters,PLOT=False,TREAT=TREAT)
                data_4d[i,j] = data_2d['matrices'][ii]
        global_max = data_4d.max()
        global_min = data_4d.min()
        
        f, ax = plt.subplots(2, 2, figsize=(5, 5), sharex='col', sharey='row')
        for i,xx_val in enumerate([xx1,xx2]):
            for j,yy_val in enumerate([yy1,yy2]):
                img = ax[j, i].imshow(data_4d[i,j], cmap=cmap, origin='lower', vmin=global_min,vmax=global_max)

                n_yticks = 5
                indices = np.array(np.linspace(0,n_steps-1, n_yticks),dtype=int)
                round_precision_y = 2 if y_min<= 0 else max(0,min(3,1-int(np.floor(np.log10(y_min)))))
                y_axis_ticks_vals = np.array([np.round(val,round_precision_y) for val in  data_2d['y_range'][indices]])
                if i==0:
                    ax[j,i].set_yticks(indices)
                    ax[j,i].set_yticklabels(list(map(str,y_axis_ticks_vals)))
                    ax[j,i].set_ylabel(y_param_name)
            
        
                n_xticks = 5
                indices = np.array(np.linspace(0,n_steps-1, n_xticks),dtype=int)
                round_precision_x = 2 if x_min<= 0 else max(0,min(3,1-int(np.floor(np.log10(x_min)))))
                x_axis_ticks_vals = np.array([np.round(val,round_precision_x) for val in data_2d['x_range'][indices]]) 
                if j==1:
                    ax[j,i].set_xticks(indices)
                    ax[j,i].set_xticklabels(list(map(str,x_axis_ticks_vals)), rotation=45, ha='center')
                    ax[j,i].set_xlabel(x_param_name)
                
                # if xx_val != yy_val:
                #     x_arr = data_2d['x_range']
                #     y_arr = data_2d['y_range']
                #     x_s = x_axis_ticks_vals[-1]
                #     y_s = y_axis_ticks_vals[-1]
                #     r_s = dict_non_default_parameters['r_s']
                #     r_r = dict_non_default_parameters['r_r']
                #     sep1 = y_arr - (1/(x_arr+1e-6)) - r_s*(x_arr-1)/(yy_val)
                #     sep2 = x_arr - (1/(y_arr+1e-6)) - r_r*(y_arr-1)/(xx_val)
                #     i_pos1 = np.where(np.where((y_min<sep1) * (sep1<y_max))[0])[0]
                #     i_pos2 = np.where(np.where((x_min<sep2) * (sep2<x_max))[0])[0]
                #     ax[j, i].plot(sep1[i_pos1], 'k--', label='sep1', lw=2)
                #     ax[j, i].plot(sep2[i_pos2], 'b-', label='sep2', lw=2)


                # if xx_val < yy_val:
                #     k1 = xx_val/yy_val
                #     y_s = y_axis_ticks_vals[-1]
                #     ax[j, i].axline((0, (1-k1)*y_s), slope=k1, color='k', linestyle='--', lw=2.2)
                    
                # if xx_val > yy_val:
                #     k1 = yy_val/xx_val
                #     x_s = x_axis_ticks_vals[-1]
                #     ax[j, i].axline(((1-k1)*x_s, 0), slope=1/k1, color='k', linestyle='--', lw=2.2)
        
        pos1 = ax[0, 0].get_position()
        pos2 = ax[1, 1].get_position()
        cbar_ax = f.add_axes([pos2.x1 + 0.02, pos2.y0, 0.02, pos1.y1 - pos2.y0])
        
        cbar = f.colorbar(img, cax=cbar_ax)        
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(length=0)
        cbar.ax.yaxis.set_tick_params(which='both',length=0)

        ax_left = f.add_axes([pos1.x0 - 0.155, pos2.y0, 0.02, pos1.y1 - pos2.y0])
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        for spine in ax_left.spines.values():
            spine.set_visible(False)
        ax_left.plot([0,0],[0,1],'k-',lw=0.5)
        ax_left.set_ylim([0,1])
        ax_left.text(-0.3,0.5,yy_param_name,ha='center',va='center',rotation=90)    
        
        ax_left.text(-0.1,0.25,str(yy2),ha='center',va='center',rotation=90)    
        ax_left.text(-0.1,0.75,str(yy1),ha='center',va='center',rotation=90)    

        ax_top = f.add_axes([pos1.x0, pos1.y1 + 0.01, pos2.x1 - pos1.x0, 0.02])
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        ax_top.plot([0,1],[0,0],'k-',lw=0.5)
        ax_top.set_xlim([0,1])
        ax_top.text(0.5,0.3,xx_param_name,ha='center',va='center')    
        ax_top.text(0.25,0.1,str(xx1),ha='center',va='center',rotation=0)    
        ax_top.text(0.75,0.1,str(xx2),ha='center',va='center',rotation=0)    
        
        #ax_top.text(0.5,-4.65,r'basic reproduction number ($R_{0}$)',ha='center',va='center')    
        
        params_varied = [x_param_name,y_param_name,xx_param_name,yy_param_name]
        string_varied = ['x','y','xx','yy']
        for key in data_2d['dict_non_default_parameters']:
            if data_2d['dict_non_default_parameters'][key] == x_param_name:
                params_varied.append(key)
                string_varied.append('x')
            if data_2d['dict_non_default_parameters'][key] == y_param_name:
                params_varied.append(key)
                string_varied.append('y')        
            if data_2d['dict_non_default_parameters'][key] == xx_param_name:
                params_varied.append(key)
                string_varied.append('xx')   
            if data_2d['dict_non_default_parameters'][key] == yy_param_name:
                params_varied.append(key)
                string_varied.append('yy')            

        text_left,text_right = params_to_text(data_2d['params'],params_varied=[x_param_name,y_param_name] if params_varied==[] else params_varied,string_varied=['x','y'] if string_varied==[] else string_varied,TREAT=TREAT)
        ax_top.text(0.1,0.4,'\n'.join(text_left),va='bottom',ha='center')
        ax_top.text(0.9,0.4,'\n'.join(text_right),va='bottom',ha='center')
        suffix = get_suffix(text_left,text_right) + '_tend'+str(t_end) 
        identifer_type = ''.join([el[0] for el in cbar_label.split(' ')])

        # for i,xx_val in enumerate([xx1,xx2]):
        #     for j,yy_val in enumerate([yy1,yy2]):
        #         if xx_val != yy_val:
        #                     k1 = xx_val/yy_val
        #                     k2 = yy_val/xx_val
        #                     x_s = (ax[j, i].get_xlim()[1]-ax[j, i].get_xlim()[0])/ax[j, i].get_xlim()[1]
        #                     y_s = (ax[j, i].get_ylim()[1]-ax[j, i].get_ylim()[0])/ax[j, i].get_ylim()[1]
        #                     img = ax[j, i].axline((0, (1-k1)*y_s), ((1-k2)*x_s, 0), color='k', linestyle='-.', lw=2)

        plt.savefig(os.path.join(folder_name,'sens_4d_'+identifer_type+'_'+suffix+'.pdf'), bbox_inches='tight')
        plt.show()
        

# %%
if __name__ == "__main__":
    dt = 1
    t_end = 600
    n_steps = 41

    # ------------------------------------------------------------------
    # 2D Sweeps
    # ------------------------------------------------------------------

    # 1. Sweep DIP (delta) vs threshold_treatment_on (Cytotoxic, d_th = 0.05)
    dict_non_default_parameters = {
        'r_s': 0.05, 'r_r': 0.05, 'r_s_treatment': 0.05, # Cytotoxic (ε = 1)
        't_s': 0.0015, 't_r': 0.0015,
        'threshold_treatment_off': 0.25, 'd_s_treatment': 0.05, # d_th = 0.05
        's0': 100, 'r0': 100
    }
    plot_2d_sensitivity('delta', 0.000125, 0.05, 'threshold_treatment_on', 0.3, 0.6, 
                        n_steps=n_steps, x_scale='log', dt=dt, t_end=t_end, 
                        dict_non_default_parameters=dict_non_default_parameters, TREAT=True)

    # 2. Sweep delta vs t_s (Cytotoxic, d_th = 0.01)
    dict_non_default_parameters = {
        'r_s': 0.05, 'r_r': 0.05, 'r_s_treatment': 0.05, 
        't_r': 0.0015,
        'threshold_treatment_on': 0.5, 'threshold_treatment_off': 0.25, 
        'd_s_treatment': 0.01, # d_th = 0.01
        's0': 100, 'r0': 100
    }
    plot_2d_sensitivity('delta', 0.000125, 0.05, 't_s', 0.000125, 0.0015, 
                        n_steps=n_steps, x_scale='log', y_scale='log', dt=dt, t_end=t_end, 
                        dict_non_default_parameters=dict_non_default_parameters, TREAT=True)


    # ------------------------------------------------------------------
    # 4D Sweeps
    # ------------------------------------------------------------------

    # 3. 4D: c vs d AND t_s vs t_r (Cytotoxic, d_th = 0.05)
    dict_non_default_parameters = {
        'r_s': 0.05, 'r_s_treatment': 0.05, 'r_r': 0.05, 
        'delta': 0.01, # Fixed moderate DIP
        'threshold_treatment_on': 0.5, 'threshold_treatment_off': 0.25, 
        'd_s_treatment': 0.05,
        's0': 100, 'r0': 100
    }
    plot_4d_sensitivity('c', 0.1, 7.4, 'd', 0.1, 7.4, 
                        't_s', 0.000125, 0.0015, 't_r', 0.000125, 0.0015, 
                        n_steps=n_steps, dt=dt, t_end=t_end, 
                        dict_non_default_parameters=dict_non_default_parameters, TREAT=True)

    # 4. 4D: c vs d AND r_s vs r_r (Cytostatic, ε = 0.33)
    dict_non_default_parameters = {
        't_s': 0.0015, 't_r': 0.0015, 
        'delta': 0.01, # Fixed moderate DIP
        'd_s_treatment': 0.0, # Cytostatic means d_th = 0
        'r_s_treatment': 0.0165, # ε * r_s = 0.33 * 0.05
        'threshold_treatment_on': 0.5, 'threshold_treatment_off': 0.25,
        's0': 100, 'r0': 100
    }
    plot_4d_sensitivity('c', 0.1, 7.4, 'd', 0.1, 7.4, 
                        'r_s', 0.00247, 0.05, 'r_r', 0.00247, 0.05, 
                        n_steps=n_steps, dt=dt, t_end=t_end, 
                        dict_non_default_parameters=dict_non_default_parameters, TREAT=True)

# %%
            
# %%
# if __name__ == "__main__":
#     dt = 1
#     t_end = 600
#     n_steps = 41

    # Steady state without therapy
    # Without transitions, when c=d, population is split equally, while c > d or vice versa leads to corresponding advantages in final proportion. Bias in r_s or r_r leads to corresponding advantages in final proportions.
    # dict_non_default_parameters =  {'t_s': 0, 't_r': 0,
    #                                 's0': 100, 'r0': 100}
    # plot_4d_sensitivity('c', 0, 5, 'd', 0, 5, 'r_s', 0.03, 0.05, 'r_r', 0.03, 0.05, t_end=1000, dict_non_default_parameters=dict_non_default_parameters, folder_name='../../../figures/no-therapy', TREAT=False)

    # With transitions and t_s = t_r, the main diagonal remains the same when r_s = r_r, but when they are unequal, transitions decrease the  advantage in terms of the final proportion for the cell type with higher growth rate.
    # dict_non_default_parameters =  {'t_s': 0.005, 't_r': 0.005,
    #                                 's0': 100, 'r0': 100}
    # plot_4d_sensitivity('c', 0, 5, 'd', 0, 5, 'r_s', 0.03, 0.05, 'r_r', 0.03, 0.05, t_end=1000, dict_non_default_parameters=dict_non_default_parameters, folder_name='../../../figures/no-therapy', TREAT=False)

    # With transitions and t_s != t_r, the main diagonal is no longer a fixed point, and higher transition rates reduce the advantage of the corresponding cell type.
    # dict_non_default_parameters =  {'r_s': 0.5, 'r_r': 0.5,
    #                                 's0': 100, 'r0': 100}
    # plot_4d_sensitivity('c', 0, 5, 'd', 0, 5, 't_s', 0.003, 0.005, 't_r', 0.003, 0.005, t_end=1000, dict_non_default_parameters=dict_non_default_parameters, folder_name='../../../figures/no-therapy', TREAT=False)
    # dict_non_default_parameters = {'r_s' : 0.05, 'r_r': 0.05, 'r_s_treatment': 0.01, 'd_s_treatment': 0.,
    #                                's0': 100, 'r0': 100,
    #                             'threshold_treatment_on': 0.5, 'threshold_treatment_off': 0.25}
    # plot_4d_sensitivity('c', 0, 5, 'd', 0, 5, 't_s', 0, 0.05, 't_r', 0, 0.05, n_steps=n_steps,x_scale='linear',dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters, folder_name='../../../figures/adaptive-therapy', TREAT=True)

    # dict_non_default_parameters = {'r_s' : 0.05, 'r_r': 0.05, 'r_s_treatment': 0.01, 'd_s_treatment': 0,
    #                                's0': 100, 'r0': 100,
    #                             'threshold_treatment_on': 0.5, 'threshold_treatment_off': 0.1}
    # plot_4d_sensitivity('c', 0, 5, 'd', 0, 5, 't_s', 0, 0.05, 't_r', 0, 0.05, n_steps=n_steps,x_scale='linear',dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters, folder_name='../../../figures/adaptive-therapy', TREAT=True)

    # 2. d_s_treatment vs delta — shows how DIP interacts with drug killing
    # dict_non_default_parameters = {'r_s':0.01,'r_s_treatment':0.01,'r_r':0.01,
    #                                't_r': 0.01, 't_s':0.01,
    #                                'threshold_treatment_off': 0.25, 'd_s_treatment': 0.1}
    # plot_2d_sensitivity('d_s_treatment', 0.01, 0.2, 'delta', 0.001, 0.1,
    #                     n_steps=n_steps, y_scale='log', dt=dt, t_end=t_end,
    #                     dict_non_default_parameters=dict_non_default_parameters, TREAT=True)


    # dict_non_default_parameters = {'r_s' : 0.05, 'r_r': 0.05,
    #                                'r_s_treatment': 'r_s', 'd_s_treatment': 0.1, 'delta':0.,
    #                                's0': 100, 'r0': 100,
    #                             'threshold_treatment_on': 0.5, 'threshold_treatment_off': 0.1}
    # plot_4d_sensitivity('c', 0, 5, 'd', 0, 5, 't_s', 0.005, 0.05, 't_r', 0.005, 0.05, n_steps=n_steps,x_scale='linear',dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters, TREAT=True)




    # dict_non_default_parameters = {'c':0.5,'d':0.5,'t_r':0,'t_s':0, 'r_s' : 0.2, 'r_r': 0.2, 'r_s_treatment' : 0.2}
    # plot_2d_sensitivity('d_s_treatment', 0.01,0.2,'threshold_treatment_on',0.25,0.5,n_steps=n_steps,x_scale='log',dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)
    
    # dict_non_default_parameters = {'r_s':0.01,'r_s_treatment':0.01,'r_r':0.01,'t_r': 0.01, 't_s':0.01, 't_s_treatment':0.01, 'threshold_treatment_off' : 0.25, 'd_s_treatment' : 0.1}
    # plot_2d_sensitivity('d_s_treatment', 0.01,0.2,'t_s_treatment',0.001,0.1,n_steps=n_steps,y_scale='log',dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)
        
    # dict_non_default_parameters = {'r_s':0.01,'r_s_treatment':0.01,'r_r':0.01,'t_r': 0.01, 't_s':0.01, 't_s_treatment':0.02, 'threshold_treatment_off' : 0.25, 'd_s_treatment' : 0.1}
    # plot_2d_sensitivity('c', 0,2,'d',0,2,n_steps=n_steps,dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)
        
    # dict_non_default_parameters = {'r_s':0.01,'r_s_treatment':0.01,'r_r':0.01, 't_s_treatment':'t_s', 'threshold_treatment_off' : 0.25, 'd_s_treatment' : 0.1}
    # plot_2d_sensitivity('t_s', 0,0.1,'t_r',0,0.1,n_steps=n_steps,dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)

    # dict_non_default_parameters = {'r_s':0.01,'r_s_treatment':'r_s', 'r_r':'r_s', 't_s_treatment':'t_s', 't_r':'t_s','threshold_treatment_off' : 0.25, 'd_s_treatment' : 0.1}
    # plot_2d_sensitivity('r_s', 0.01,0.1,'t_s',0.01,0.2,n_steps=n_steps,dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)
    
    # #The smaller the difference between therapy start and stop, the more therapy onsets we see
    # dict_non_default_parameters = {'d':'c','r_s':0.01,'r_s_treatment':'r_s', 'r_r':'r_s', 't_s':0.01,'t_s_treatment':'t_s', 't_r':'t_s','threshold_treatment_off' : 0.25, 'd_s_treatment' : 0.1}
    # plot_2d_sensitivity('threshold_treatment_off', 0.1,0.3,'threshold_treatment_on',0.4,0.6,n_steps=n_steps,dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)

    # #Presence/absence of competition really doesn't seem to have much of an effect    
    # dict_non_default_parameters = {'r_s':0.01,'r_s_treatment':'r_s', 'r_r':'r_s', 't_s_treatment':'t_s', 't_r':'t_s','threshold_treatment_off' : 0.25, 'd_s_treatment' : 0.1}
    # plot_4d_sensitivity('r_s', 0.01,0.05,'t_s',0.01,0.05,'c',0,2,'d',0,2,n_steps=n_steps,dt=dt,t_end=t_end,dict_non_default_parameters=dict_non_default_parameters)

# %%