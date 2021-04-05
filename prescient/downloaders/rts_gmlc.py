#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

'''
Script to download the RTS-GMLC repository from GitHub
and do some basic population of prescient scripts
'''
import os

this_file_path = os.path.dirname(os.path.realpath(__file__))
rts_download_path = os.path.realpath(os.path.join(this_file_path, os.path.normcase('../../downloads/rts_gmlc')))

def populate_input_data():
    from prescient.downloaders.rts_gmlc_prescient.process_RTS_GMLC_data import create_timeseries
    from prescient.downloaders.rts_gmlc_prescient.rtsgmlc_to_dat import write_template

    create_timeseries(rts_download_path)

    print("\nWriting dat template file...")
    write_template(rts_gmlc_dir=os.path.join(rts_download_path,'RTS-GMLC'),
                   file_name= os.path.join(rts_download_path,os.path.normcase('templates/rts_with_network_template_hotstart.dat')))

def copy_templates():
    import shutil
    cur_path = os.path.join(this_file_path,os.path.normcase('rts_gmlc_prescient/runners'))
    new_path = rts_download_path

    if os.path.exists(new_path):
        return

    shutil.copytree(cur_path, new_path)

def download(branch='HEAD'):
    '''
    Clones the RTS-GMLC repository from GitHub at https://github.com/GridMod/RTS-GMLC

    Parameters
    ----------
    branch : str (optional) 
        The commit tag to check out after cloning.  Default is "HEAD".
    '''
    
    cur_path = os.getcwd()

    if branch == 'HEAD':
        branch_name = ''
    else:
        branch_name = '_'+branch

    rtsgmlc_path = os.path.realpath(os.path.join(rts_download_path,'RTS-GMLC'+branch_name))

    if os.path.exists(rtsgmlc_path):
        print('RTS-GMLC already downloaded to {0}. If you would like re-download it, delete the directory {0}.'.format(rtsgmlc_path))
        return

    print('Downloading RTS-GMLC into '+rtsgmlc_path)

    url = 'https://github.com/GridMod/RTS-GMLC.git'

    clone_cmd = 'git clone -n '+url+' '+rtsgmlc_path
    ret = os.system(clone_cmd)
    if ret:
        raise Exception('Issue cloning RTS-GMLC repository; see message above.')

    os.chdir(rtsgmlc_path)

    checkout_cmd = 'git checkout '+branch
    ret = os.system(checkout_cmd)
    if ret:
        raise Exception('Issue checking out {}; see message above.'.format(branch))

    os.chdir(cur_path)


if __name__ == '__main__':
    download()
    copy_templates()
    populate_input_data()
    print('Set up RTS-GMLC data in {0}'.format(rts_download_path))
