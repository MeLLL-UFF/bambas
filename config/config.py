import json
from configparser import ConfigParser


with open("config_fields.json", mode="r") as file:
    FIELD_VALIDATION = json.load(file)

'''
Accepts a path to a config file
Returns a dict with configs
'''
def get_config(config_path:str)->dict:
    parser = ConfigParser()
    parser.read(config_path)
    
    config = {}

    for section in parser:
        config[section] = {}
        for field in parser[section]:
            print(field)
            config[section][field] = validate_field(section, 
                                                    field, 
                                                    parser[section][field])
    
    return config

'''
Receives a field and its value
Returns the field converted to its type
'''
def validate_field(section: str, field:str, value:str):
    
    try:
        accepted_values = FIELD_VALIDATION[section][field]["accepted_values"]
        field_type = FIELD_VALIDATION[section][field]["type"]
    except KeyError:
        print("field not registered in the config_fields.json file, it will be handled as string")
        return value

    if accepted_values != [] and value not in accepted_values:
        raise ValueError(
            "Config field {0} must be one of these values {1}, it was {2}"\
            .format(field, accepted_values, value))
    
    if field_type == "str":
        return str(value)
    elif field_type == "int":
        return int(value)
    elif field_type == "bool":
        if value=="True": 
            return True
        else: 
            return False
    elif field_type == "list":
        return value.replace('[','')\
                    .replace(']','')\
                    .replace(' ','')\
                    .split(',')
    else:
        return value


if __name__ == "__main__":
    
    print(get_config("config.cfg"))