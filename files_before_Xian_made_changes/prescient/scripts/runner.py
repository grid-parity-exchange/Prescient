#!/usr/bin/python3
"""
runner.py

This program will run python programs that have attached configuration (.txt)
files. The text file should start with the line "command/exec program.py",
where program.py stands for the python file you want to run and a list of all
the options, each in their own line, to run the program with.
The program ignores comments marked with the "#" symbol.
"""

import sys
import os
import subprocess
import importlib

def parse_line(option_string):
    orig = option_string
    opts = []
    while option_string:
        q_index = option_string.find('"')
        if q_index == -1:
            option_string = option_string.replace('=', ' ')
            option_string = option_string.replace('|', os.sep)
            opts.extend(option_string.split())
            break
        else:
            end_quote = option_string.find('"', q_index+1)
            if end_quote == -1:
                raise RuntimeError("Quote opened but no end quote in line '{}'"
                                   .format(option_string))
            arg = option_string[q_index+1:end_quote]
            keyword = option_string[:q_index].replace('=', ' ')
            keyword = keyword.replace('|', os.sep).strip()
            opts.extend([keyword, arg])
            option_string = option_string[end_quote+1:]
    return opts

def parse_commands(filename):
    """Accepts a configuration filename and
    parses out the program name and the options
    from the file and returns them.
    """
    options = []
    with open(filename) as f:
        for line in f:
            if line.startswith("command/exec"):
                _, program = line.split()
                break
        else:
            print("No Program name found")
            print("Include a line that starts with 'command/exec'")
            print("followed by the program name")
            sys.exit()

        for i, line in enumerate(f):
            if line.startswith('--'):
                options.extend(parse_line(line))
    return program, options


def main():
    if len(sys.argv) != 2:
        print("You must list the file with the program configurations")
        print("after the program name")
        print("Usage: runner.py config_file")
        sys.exit()

    config_filename = sys.argv[1]
    if not(os.path.isfile(config_filename)):
        print("{} is not a file or does not exist".format(config_filename))
    script, options = parse_commands(config_filename)

    # We pass shell=True for Windows machines, this somehow makes it properly
    # pass through command line arguments.
    if sys.platform.startswith('win'):
        subprocess.call([script] + options, shell=True)
    else:
        subprocess.call([script] + options)


if __name__ == '__main__':
    main()
