'''Functions for parsing args.'''

from ast import literal_eval
import copy
import os
import yaml

class CfgNode(dict):
    '''
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    '''

    def __init__(self, init_dict=None, key_list=None):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, value in init_dict.items():
            if isinstance(value, dict):
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(value, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(seq_, num_spaces):
            seq = seq_.split("\n")
            if len(seq) == 1:
                return seq_
            first = seq.pop(0)
            seq = [(num_spaces * " ") + line for line in seq]
            seq = "\n".join(seq)
            seq = first + "\n" + seq
            return seq

        r = ""
        seq = []
        for k, value in sorted(self.items()):
            seperator = "\n" if isinstance(value, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(value))
            attr_str = _indent(attr_str, 2)
            seq.append(attr_str)
        r += "\n".join(seq)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
    '''Load from config files.'''

    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, value in cfg_from_file[key].items():
            cfg[k] = value

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg, cfg_list):
    '''Merge configs from a list.'''

    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg


def _decode_cfg_value(v):
    '''Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    '''
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, full_key):
    '''Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    '''

    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type or original is None:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )
