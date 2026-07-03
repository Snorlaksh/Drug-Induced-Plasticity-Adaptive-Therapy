import os
import pickle
import numpy as np


def save_data(data, filename, folder='data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data successfully saved to {file_path}")
    
    
def load_data(filename, folder='data'):
    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
    

def params_to_text(params, params_varied=[], string_varied='*', TREAT=False):
    if type(string_varied) == str:
        string_varied = [string_varied] * len(params_varied)
    assert len(string_varied) == len(params_varied), "a string must be provided or a list of strings of equal length as params_varied"
    
    # Copy to avoid modifying the original dictionary during execution
    local_params = params.copy()
    for i, parameter in enumerate(params_varied):
        local_params[parameter] = string_varied[i]
        
    # ---------------------------------------------------------
    # Extract new parameters (with fallbacks to legacy names)
    # ---------------------------------------------------------
    mu_s = local_params.get('mu_s', local_params.get('r_s'))
    mu_r = local_params.get('mu_r', local_params.get('r_r'))
    alpha_rs = local_params.get('alpha_rs', local_params.get('c'))
    alpha_sr = local_params.get('alpha_sr', local_params.get('d'))
    K = local_params.get('K')
    t_s0 = local_params.get('t_s0', local_params.get('t_s'))
    t_r0 = local_params.get('t_r0', local_params.get('t_r'))
    
    # Dynamically calculate treatment values if using legacy inputs
    delta = local_params.get('delta', 0.0)
    if 'delta' not in local_params and 't_s_treatment' in local_params:
        if type(local_params['t_s_treatment']) != str and type(t_s0) != str:
            delta = local_params['t_s_treatment'] - t_s0
            
    xi = local_params.get('xi', 1.0)
    if 'xi' not in local_params and 'r_s_treatment' in local_params:
        if type(local_params['r_s_treatment']) != str and type(mu_s) != str:
            xi = local_params['r_s_treatment'] / mu_s if mu_s != 0 else 1.0
            
    d_h = local_params.get('d_h', 0.0)
    if 'd_h' not in local_params and 'd_s_treatment' in local_params:
        d_h = local_params.get('d_s_treatment', 0.0)
        
    threshold_treatment_on = local_params.get('threshold_treatment_on', np.nan)
    threshold_treatment_off = local_params.get('threshold_treatment_off', np.nan)

    text_left = []
    text_right = []
    
    # 1. Growth Rates
    if mu_s == mu_r:
        text_left.append(r'$\mu_s = \mu_r = $'+str(mu_s))
    else:
        text_left.append(r'$\mu_s = $'+str(mu_s)+r', $\mu_r = $'+str(mu_r))
        
    # 2. Competition
    if alpha_rs == alpha_sr:
        text_left.append(r'$\alpha_{rs} = \alpha_{sr} = $'+str(alpha_rs)+r', $K = $' + str(K))
    else:
        text_left.append(r'$\alpha_{rs} = $'+str(alpha_rs)+r', $\alpha_{sr} = $'+str(alpha_sr)+r', $K = $' + str(K))
        
    # 3. Transitions & Plasticity
    if t_s0 == t_r0:
        text_left.append(r'$t_{s0} = t_{r0} = $'+str(t_s0)+r', $\delta = $'+str(delta))
    else:
        text_left.append(r'$t_{s0} = $'+str(t_s0)+r', $t_{r0} = $'+str(t_r0)+r', $\delta = $'+str(delta))
                
    # 4. Therapy Thresholds
    text_right.append('therapy start'+r'$ = $'+str(threshold_treatment_on)+'K')
    text_right.append('therapy stop'+r'$ = $'+str(threshold_treatment_off)+'K')
    
    # 5. Drug Efficacy
    text_right.append(r'$\xi = $'+str(xi)+r', $d_h = $'+str(d_h))

    return text_left, text_right


def get_suffix(text_left, text_right):
    suffix = ','.join(text_left) + ',' + ','.join(text_right)
    
    # 1. Safely convert LaTeX variables to file-friendly alphanumeric strings
    suffix = suffix.replace(r'\mu_s', 'mus').replace(r'\mu_r', 'mur')
    suffix = suffix.replace(r'\alpha_{rs}', 'alphars').replace(r'\alpha_{sr}', 'alphasr')
    suffix = suffix.replace(r't_{s0}', 'ts0').replace(r't_{r0}', 'tr0')
    suffix = suffix.replace(r'\delta', 'delta').replace(r'\xi', 'xi').replace(r'd_h', 'dh')
    
    # 2. Standard formatting cleanup
    suffix = suffix.replace(' ', '')
    suffix = suffix.replace('_', '')
    suffix = suffix.replace('{', '').replace('}', '').replace('^', '')
    suffix = suffix.replace(',', '_')
    suffix = suffix.replace('=', '_')
    suffix = suffix.replace('therapy', 'th')
    suffix = suffix.replace('$', '')
    
    return suffix