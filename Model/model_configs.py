

def main(config_set=1):

    if config_set == 1:
        d = {
            'loss_func': 'rmsm',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'OnePoleAttOnly', 'cond': False},
            'make_up': {'type': 'Static'}
        }

    elif config_set == 2:
        d = {
            'loss_func': 'rmsm',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'OnePoleAttRel', 'cond': False},
            'make_up': {'type': 'Static'}
        }

    elif config_set == 3:
        d = {
            'loss_func': 'rmsm',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'TimeVaryOP', 'cond': False, 'hidden_size': 4, 'rec': 'rnn'},
            'make_up': {'type': 'Static'}
        }

    elif config_set == 4:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'hk'},
            'gain_smooth': {'type': 'OnePoleAttRel', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 4}
        }
    elif config_set == 5:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'hk'},
            'gain_smooth': {'type': 'OnePoleAttOnly', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 8}
        }

    elif config_set == 6:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'hk'},
            'gain_smooth': {'type': 'OnePoleAttRel', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 8}
        }

    elif config_set == 7:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'TimeVaryOP', 'cond': False, 'hidden_size': 4, 'rec': 'rnn'},
            'make_up': {'type': 'GRU', 'hidden_size': 8}
        }
    elif config_set == 8:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'OnePoleAttOnly', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 8}
        }
    elif config_set == 9:
        d = {
            'loss_func': 'ESR',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'FixedOnePole', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 8}
        }
    elif config_set == 10:
        d = {
            'loss_func': 'rmsm',
            'data_set': 'LA-2A',
            'data_dir': 'dataset',
            'static_comp': {'type': 'sk'},
            'gain_smooth': {'type': 'FixedOnePole', 'cond': False},
            'make_up': {'type': 'GRU', 'hidden_size': 32}
        }
    else:
        print('no config')

    return d