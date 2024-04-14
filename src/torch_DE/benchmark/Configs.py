


def get_config_dict(turn_on_all = False):
    config = {option:turn_on_all for option in ['Sampling','RWF','GradNorm','SDF']}
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
    

def independent_runs():
    '''
    Create configs where we only turn one setting on at a time
    '''
    config = get_config_dict(turn_on_all=False)

    configs = [get_config_dict for _ in range(len(config.keys()))]

    for i,key in enumerate(config.keys()):
        configs[i][key] = True

    return {f'Only_{key}': config for config,key in zip(configs,configs.keys())}


def excluded_runs():
    '''
    Create configs where we have all settings turned on except for one setting
    '''
    config = get_config_dict(turn_on_all=True)

    configs = [get_config_dict for _ in range(len(config.keys()))]

    for i,key in enumerate(config.keys()):
        configs[i][key] = False

    return {f'No_{key}': config for config,key in zip(configs,configs.keys())}




