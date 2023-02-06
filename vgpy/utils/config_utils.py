import os
from configparser import ConfigParser

def get_config(filePath):
    conf = ConfigParser()
    if os.path.isfile(filePath):
        conf.read(filenames=filePath, encoding='utf-8')
    else:
        print('config path not found:', filePath)
        print('return a empty config')
    return conf


def save_config(filePath, section, option, value):
    config = get_config(filePath)

    if not config.has_section(section):
        config.add_section(section)

    if not config.has_option(section, option):
        config.set(section, option, '')

    config[section][option] = value
    with open(str(filePath), 'w', encoding='utf-8') as f:
        config.write(f)


def add_config_option(filePath, section, option):
    config = get_config(filePath)

    if not config.has_option(section, option):
        config.set(section, option, '')

    with open(str(filePath), 'w', encoding='utf-8') as f:
        config.write(f)


def remove_config_option(filePath, section, option):
    config = get_config(filePath)
    config.remove_option(section, option)

    with open(filePath, 'w', encoding='utf-8') as f:
        config.write(f)

def remove_config_section(filePath, section):
    config = get_config(filePath)
    config.remove_section(section)

    with open(filePath, 'w', encoding='utf-8') as f:
        config.write(f)