# @Time : 2021/6/22 16:46
# @Author : Richard FANG
# @File : global_var.py.py 
# @Software: PyCharm


def _init():  # Initialize global dictionary
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    # Define a global variable
    _global_dict[key] = value


def get_value(key):
    # Retrieve a global variable; print an error message if it does not exist
    try:
        return _global_dict[key]
    except:
        print(f'Failed to retrieve key "{key}"\r\n')