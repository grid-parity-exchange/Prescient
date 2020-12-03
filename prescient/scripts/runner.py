#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

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
import threading
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

def run(config_filename, **kwargs):
    '''
    Parameters
    ----------
    config_filename : str
        Path to Prescient txt file
    kwargs :
        Additional arguments to subprocess.Popen

    Returns
    -------
    Result from subprocess.Popen

    '''

    if not(os.path.isfile(config_filename)):
        print("{} is not a file or does not exist".format(config_filename))
    script, options = parse_commands(config_filename)

    ## we append an *.exe to the script so windows is happy
    if sys.platform.startswith('win'):
        script = script+'.exe'
    os.environ['PYTHONUNBUFFERED'] = '1'

    proc = subprocess.Popen([script] + options,
                            stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            **kwargs)

    def write_output(stream, dest):
        for line in stream:
            print(line.decode('utf-8'), end='', file=dest)
        stream.close()

    def run_process(process):
        process.wait()

    tout = threading.Thread(target=write_output, args=(proc.stdout, sys.stdout))
    terr = threading.Thread(target=write_output, args=(proc.stderr, sys.stderr))
    tproc = threading.Thread(target=run_process, args=(proc,))

    tout.start()
    terr.start()
    tproc.start()

    tproc.join()
    tout.join()
    terr.join()

    return proc

def main():
    if len(sys.argv) != 2:
        print("You must list the file with the program configurations")
        print("after the program name")
        print("Usage: runner.py config_file")
        sys.exit()

    config_filename = sys.argv[1]
    run(config_filename)

if __name__ == '__main__':
    main()
