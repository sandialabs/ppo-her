from flatdict import FlatDict
import os
from flatten_dict import flatten
import json
from ppo_her.base.config import DEFAULT_CONFIG
import git
from warnings import warn
from copy import deepcopy

'''
    Get a string name from the config dictionary
    Inputs:
        config (dict) - the config dictionary
    Returns:
        name (str) - the string version of the name
'''
def get_name(config):

    # Convert config to flattened dict
    flat_dict = FlatDict(config)
    back_2_dict = dict(flat_dict)

    # Convert default config to flattened dict
    default_config = deepcopy(DEFAULT_CONFIG)
    flat_default = FlatDict(default_config)
    default_2_dict = dict(flat_default)

    # Remove fields that are the same as the default
    partial_dict = {}
    for key in back_2_dict.keys():
        if key in list(default_2_dict.keys()):
            if back_2_dict[key] == default_2_dict[key]:
                continue

            # Ignore if it is a custom object and not a built-in type
            if back_2_dict[key].__class__.__module__ != 'builtins':
                continue

        partial_dict[key] = back_2_dict[key]

    # Create the name
    name = ''    
    for num, (key, value) in enumerate(partial_dict.items()):
        if num > 0:
            name += '|'
        name += str(key).split(':')[-1]#[:5]
        name += '-'
        name += str(value)[:15]

    
    if name == '':
        name = 'default'

    return name

'''
    Get the experiment from the name of the file being passed
    Typical usage:
        experiment_name = get_experiment_name(__file__)
    Inputs:
        file (str) - the name of the file, which will typically
            be derived by using __file__
    Returns:
        experiment_name (str) - the name of the experiment
'''
def get_experiment_name(file_name):
    file_path = os.path.dirname(os.path.realpath(file_name))
    experiment_name = os.path.split(file_path)[1]
    return experiment_name


'''
    Get the folder where different data types are/should be stored
    Inputs:
        data_type (str) - the type of data that we are trying to store
    Returns:
        path (str) - the path to the parent folder of the data
'''
def get_folder(data_type):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    split_directory = current_directory.split(os.sep)
    top_directory = os.path.join(*split_directory[:-2])

    path_dict = {
        'images': os.path.join(top_directory, 'images'),
        'summary_data': os.path.join(top_directory, 'summary_data'),
        'tb_logs': os.path.join(top_directory, 'tb_logs'),
        'ray_results': os.path.join(top_directory, 'ray_results'),
        'experiments': os.path.join(top_directory, 'meher', 'experiments'),
        'stats': os.path.join(top_directory, 'stats'),
        'tex': os.path.join(top_directory, 'tex'),
        'models': os.path.join(top_directory, 'models'),
        'configs': os.path.join(top_directory, 'configs'),
        'gifs': os.path.join(top_directory, 'gifs'),
    }
    
    assert data_type in list(path_dict.keys())
    return (os.sep + path_dict[data_type])


'''
    If the path for saving a file does not exist, create it
    Inputs:
        folder_path (str) - the path of the folder where the file will be saved
'''
def safe_save(folder_path):
    folders = folder_path.split(os.sep)
    num_folders = len(folders)
    folders[0] = os.sep + folders[0]

    for folder_num in range(num_folders):
        temp_folder = os.path.join(*folders[:folder_num + 1])
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

'''
    Save a config dict as json
    Inputs:
        config (dict) - the config to save
        experiment_name (str) - the name of the current experiment
'''
def save_config(config, experiment_name):

    # Create the folder, if needed
    config_save_name = os.path.join(get_folder('configs'), experiment_name)
    safe_save(config_save_name)

    # Determine the full file name
    config_name = get_name(config)
    config_save_name = os.path.join(config_save_name, config_name + '.json')

    # Flatten the config
    flat_config = flatten(config, reducer='dot')

    # Make sure that objects are stored as strings
    for key, value in flat_config.items():
        if (value.__class__.__module__ != 'builtins') or (isinstance(value, type)):
            flat_config[key] = '__' + str(value) + '__'

    with open(config_save_name, 'w') as json_file:
        json.dump(flat_config, json_file)


'''
    Get the current git commit and any changes
    Returns:
        commit (str) - the last git commit
        changes (str) - changes since the last commit
        has_change (bool) - if the repo has been changed since the last commit
'''
def get_git():
    repo = git.Repo(search_parent_directories=True)
    
    try:
        commit = repo.head.object.hexsha

        changes = repo.git.diff(repo.commit())
        if changes == '':
            changes = 'None!'

    except:
        changes = 'None'
        commit = "000000"
        warn('No commits yet.  Return null (000000)')

    return commit, changes

'''
    Get the full experiment name
    Inputs:
        model_config (dict) - the config dictionary - just the model part
    Returns
        tensorboard_log (str) - the path for results
'''
def get_tensorboard_name(model_config):
    tb_logs = get_folder('tb_logs')
    tensorboard_log = os.path.join(
        tb_logs,
        model_config['experiment_name'],
        'sb3_results',
        model_config['config_name'], 
    )
    return tensorboard_log


'''
    Test if code works
'''
if __name__ == '__main__':
    print(get_folder('images'))
    print(get_folder('summary_data'))
    print(get_folder('tb_logs'))
    safe_save(
        os.path.join(get_folder('tb_logs'), 'test_exp')
    )
