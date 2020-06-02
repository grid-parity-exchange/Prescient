#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import os

import pandas as pd

def prescient_output_diff(results_dir_a, results_dir_b, numeric_fields, tolerances):
    """Compares two sets of Prescient simulator output files to determine where differences exceed specified tolerances. 
    Returns corresponding difference files in [results_dir_a]_[results_dir_b]_diff.
    
    :param results_dir_a: The root directory of the first set of results.
    :param results_dir_a: str
    :param results_dir_b: The root directory of the second set of results.
    :param results_dir_b: str
    :param numeric_fields: Dictionary containing list of column names of numeric-valued fields for each simulator output file.
    :param numeric_fields: dict
    :param tolerances: Dictionary containing difference tolerance values for each numeric-valued field for each simulator output file.
    :param tolerances: dict
    """
    # Compiles the names of output files based on results dir a and collects the DataFrames for each.
    csv_files_results_a = []
    dfs_results_a = {}
    files_to_exclude = {'Options.csv', }

    has_difference = False

    for csv_file in os.scandir(results_dir_a):
        if csv_file.name[-3:] == 'csv' and csv_file.name not in files_to_exclude:
            csv_file_root = csv_file.name[:-4]
            csv_file_path = csv_file.path

            csv_files_results_a.append(csv_file)
            dfs_results_a[csv_file_root] = pd.read_csv(csv_file_path)
    
    # Compiles the corresponding analog DataFrames for results_dir_b.
    dfs_results_b = {}

    for csv_file in csv_files_results_a:
        csv_file_root = csv_file.name[:-4]
        csv_path_results_b = os.path.join(results_dir_b, csv_file.name)

        try:
            df = pd.read_csv(csv_path_results_b)
        except FileNotFoundError:
            # No equivalent csv file found for results set b.
            raise(FileNotFoundError('No matching file for {0} found in the results_dir_b directory.'.format(csv_file.name)))
        else:
            dfs_results_b[csv_file_root] = df
    
    # Static definitions for numeric fields in each output file and the diff tolerance for each. 
    # TODO: Why is there extra whitespace on certain column names?

    # Function used to check differences between results sets.
    def _check_all_differences(row):
        """Compares all numerical fields and returns True if any numeric column's difference exceeds its tolerance."""
        diff = False

        numeric_field_tolerances = tolerances.get(csv_file_root, {})

        for field in numeric_fields[csv_file_root]:
            # Use some default value if no tolerance for the field is specified.
            tol = numeric_field_tolerances.get(field, 1e-2)

            if abs(row['{0}_A'.format(field)] - row['{0}_B'.format(field)]) > tol:
                # "any" -> short-circuit
                diff = True
                return diff

        return diff

    # Get the root directory name of each set of results.
    results_a_root = os.path.split(results_dir_a)[-1]
    results_b_root = os.path.split(results_dir_b)[-1]

    # Create the diff output directory if necessary. This goes in the same parent directory as results_dir_a.
    output_dir = os.path.join(
        os.path.split(results_dir_a)[0], 
        '{0}_{1}_diff'.format(results_a_root, results_b_root)
    )
    os.makedirs(output_dir, exist_ok=True)

    for csv_file in csv_files_results_a:
        # Iterate over each output file and generate a difference report.
        csv_file_root = csv_file.name[:-4]

        df_a = dfs_results_a[csv_file_root]
        df_b = dfs_results_b[csv_file_root]

        # Skip if DataFrame (a) is empty.
        if df_a.empty:
            continue

        # Join corresponding DataFrames and determine if any numeric column has a difference.
        df_joined = df_a.join(df_b, how='left', lsuffix='_A', rsuffix='_B')
        df_joined['has_difference'] = df_joined.apply(lambda row: _check_all_differences(row), axis=1)

        # Filter flagged differences.
        df_diff_report = df_joined.loc[df_joined['has_difference'] == True]

        if not df_diff_report.empty:
            # At least one difference exceeded its tolerance level.
            has_difference = True
            print('df_diff_report')
            print(csv_file)

        # Output diff results to csv files.
        df_diff_report.to_csv(os.path.join(output_dir, csv_file.name))
    
    return has_difference
