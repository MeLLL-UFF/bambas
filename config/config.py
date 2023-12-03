from configparser import ConfigParser


'''
Accepts a path to a config file
Returns a dict with configs
'''
def get_config(config_path:str)->dict:
    parser = ConfigParser()
    parser.read(config_path)
    return parser

if __name__ == "__main__":
    cfg = get_config("config.cfg")
    for section in cfg:
        print(cfg[section])
        for field in cfg[section]:
            print(cfg[section][field])