


def get_config_dict(turn_on_all = False):
    config = {option:turn_on_all for option in ['R3_Sampling','RWF','GradNorm','SDF',]}
    return config

def config_run(*options,all_on = True):
    '''
    Set up a config based on options given.
    If all_on = False then all options by default are turned off and specifying an option turns that on
    If all_on = True then all options by default are turned on and specifying an option turns that option off
    '''
    
    if all_on:
        config = get_config_dict(all_on)
        for option in options:
            if option in config.keys():
                config[option] = False
    else:
        
        config = get_config_dict(turn_on_all = not all_on )
        for option in options:
            if option in config.keys():
                config[option] = True
    return config
    

def independent_runs(*,exclude_options:list = None):
    '''
    Create configs where we only turn one setting on at a time as well as a config where all settings are turned off

    Current Keys:
        - 'R3_Sampling'
        - 'RWF'
        - 'GradNorm'
        - 'SDF'
    '''
    config = get_config_dict(turn_on_all=False)

    configs = [get_config_dict(False) for _ in range(len(config.keys()))]

    keys = tuple(config.keys())
    for i in range(len(configs)):
        key = keys[i]
        configs[i][key] = True

    if exclude_options is None:
        exclude_options = []
    elif isinstance(exclude_options,str):
        exclude_options = [exclude_options]

    configs_dict = {f'Only_{k}': c for c,k in zip(configs,keys) if k not in exclude_options}
    configs_dict['All_Off'] = config
    
    return configs_dict


def excluded_runs(*,exclude_options = None):
    '''
    Create configs where we have all settings turned on except for one setting

    Current Keys:
    - 'R3_Sampling'
    - 'RWF'
    - 'GradNorm'
    - 'SDF'

    '''
    config = get_config_dict(turn_on_all=True)

    configs = [get_config_dict(turn_on_all=True) for _ in range(len(config.keys()))]

    keys = tuple(config.keys())
    for i in range(len(configs)):
        key = keys[i]
        configs[i][key] = False

    configs_dict = {f'Only_{k}': c for c,k in zip(configs,keys) if k not in exclude_options}
    configs_dict['All_On'] = config
    
    return configs_dict



