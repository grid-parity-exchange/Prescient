#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

## some useful function decorators for building these dynamic models
from functools import wraps

def add_model_attr(attr, requires = {}):
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            ## tag this function in the model with the appropriate attribute
            model = args[0]
            if hasattr(model, attr):
                raise Exception("Exception adding %s! Model already has %s %s! You may only add one type of %s!"%(func.__name__, attr, getattr(model,attr), attr)) 
            # this checks to see if the required components were already added
            for base_attr in requires:
                if not hasattr(model, base_attr):
                    raise Exception("Exception adding %s! %s requires some %s to be added first!"%(func.__name__, func.__name__, base_attr)) 
                ## None in this context means there is no specific requirement
                if requires[base_attr] is None:
                    continue
                if getattr(model, base_attr) not in requires[base_attr]:
                    raise Exception("Exception adding %s! %s requires one of: "%(func.__name__, func.__name__) + ", ".join(requires[base_attr]) + ", to be added first.")
            setattr(model, attr, func.__name__)
            return func(*args, **kwds)
        return wrapper
    return actual_decorator

