from optparse import Option

def add_custom_commandline_option(option: Option) -> None:
    '''
    To add custom command-line options to Prescient, create a file that
    calls this function and include that file as a plugin on the command
    line (--plugin=my_file).
    '''
    from .internal import active_parser
    active_parser.add_option(option)
