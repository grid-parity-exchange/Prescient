# This file just identifies the data type used to pass options within prescient.
# If we ever change options parsers, we can change the options type once here instead of
# changing it everywhere.

import optparse 

Options = optparse.Values

