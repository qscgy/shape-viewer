import yaml

DEFAULT = {
    'surface': {'equation': "1 - x*x - y*y - 2.5*exp(-(x*x/4 + y*y))",
                'scalars': 'gaussian_k'},
    'lims': {'xmin':-1,
                       'ymin':-1,
                       'xmax':1,
                       'ymax':1,
                       'steps': 61},
    'gaussmap': False,
    'normals': {'plot': False,
                          'color': 'green'},
    'principal_dirs': {'plot': False,
                       'colors': ['black', 'orange'],
                       'draw_curves': False},
    'asymptotic_dirs': {'plot': False,
                        'colors': ['red', 'blue'],
                        'draw_curves': False},
    'parabolic_curves': False,
    'ridges': False,
    'flecnodes': False
}

def _set_config_defaults(cfg, default=DEFAULT):
    for dk in default:
        if dk not in cfg:
            cfg[dk] = default[dk]
        elif isinstance(default[dk], dict):
            _set_config_defaults(cfg[dk], default=default[dk])