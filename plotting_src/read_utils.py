
from plotting_src.header_settings import *
import pandas as pd

def mk_missing_dir(directory, depth=0):
    if depth>len(directory.split('/'))-1:
        depth = len(directory.split('/'))-1
    while depth>0:
        dir_temp = os.path.join(*directory.split('/')[:-depth])
        depth-=1
        if not os.path.exists(dir_temp):
            os.mkdir(dir_temp) 
    if not os.path.exists(directory):
            os.mkdir(directory)
    return directory

def fetch_raw_file(prediction_filename, raw_input_dir):
    entries = os.listdir(raw_input_dir)
    for e in entries:
        sample = '_'.join(prediction_filename.split('/')[-1].split('_')[:2])
        if sample in e:
            return os.path.join(raw_input_dir,e)
    

def is_emptystr(s):
    return len(s.strip())==0

def header_lines(filename, until="cell_id"):
    with open(filename,'r') as fin:
        for i, line in enumerate(fin):
            if line.startswith(until):
                return i
    return None



def sort_by_condition(files):
    first = []
    for prom in promoters:
        for file in files:
            if prom in file:
                first.append(file)
    second = []
    for condition in conditions:
        for file in first:
            if condition+'_' in file:
                second.append(file)
    
    return second

def sort_by_promoter(files):
    first = []
    for condition in conditions:
        for file in files:
            if condition+'_' in file:
                first.append(file)
    second = []
    for prom in promoters:
        for file in first:
            if prom in file:
                second.append(file)
    return second
    

def get_promoter(filename):
    for prom in promoters:
        if prom in filename:
            return prom
    return None


def get_condition(filename):
    for condition in conditions:
        if condition in filename:
            return condition
    return None

def get_year(filename):
    for year in years:
        if '_' + year in filename:
            return year
    return None


def get_file_by_condition_promoter(condition, promoter,filenames, all_files=True):
    files = []
    for file in filenames:
        if promoter == get_promoter(file) and condition == get_condition(file):
            files.append(file)
    if all_files:
        return files
    else:
        return sorted(files)[0]

def filter_files(files, filter_key):
    out = []
    for file in files:
        if filter_key in file:
            out.append(file)
    return out


def find_file(files, *args):
    for file in files:
        fine = True
        for arg in args:
            if not arg in file:
                fine = False
        if fine:
            return file
    
    
def find_out_dirs(file_path):
    outs = []
    enties = os.listdir(file_path)
    for e in enties:
        if e.endswith('_out'):
            outs.append(os.path.join(file_path , e))
    return outs


def find_files(file_path, ending='final.csv'):
    finals = []
    enties = os.listdir(file_path)
    for e in enties:
        if e.endswith(ending):
            finals.append(os.path.join(file_path , e))
    return sorted(finals)


def find_files_in_outs(file_path, ending='final.csv'):
    outs = find_out_dirs(file_path)
    finals = []
    for out in outs:
        enties = os.listdir(out)
        for e in enties:
            if e.endswith(ending):
                finals.append(os.path.join(out , e))
    return sorted(finals)
    
    
def sort_by_key(files, sort_key):
    out = []
    for sk in sort_key:
        for f in files:
            if sk in f:
                out.append(f)
    return out



def get_input_files(directory, keyword=None, ext=".csv"):
    entries = os.listdir(directory)
    final_files = []
    if keyword == None:
        for e in entries:
            if e.endswith(ext):
                final_files.append(os.path.join(directory,e))
    else:
        for e in entries:
            if e.endswith(ext) and keyword in e:
                final_files.append(os.path.join(directory,e))   
    return sorted(final_files)

def read_ggp_csv(filemane):
    return pd.read_csv(filemane, 
                       skiprows=header_lines(filemane), 
                       dtype={"parent_id":str, "cell_id":str})


def read_final_params(filename):
        parameters_arr = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=header_lines(filename, until="10")+1)
        param_dict = {}
        for param in parameters_arr[1:]:
            if param[-1] != '':
                param_dict[param[1]] = float(param[-1])
            else:
                param_dict[param[1]] = float(param[3])
        return param_dict


    
def read_parameter_file(filename):
    parameter_settings ={}
    with open(filename,'r') as fin:
        for i, line in enumerate(fin):
            if  not line.startswith("#"):
                parameter = line.strip().split('=')[0].strip()
                param_vals = line.strip().split('=')[1].split(',')
                if len(param_vals)>1:
                    parameter_settings[parameter] =["free", float(param_vals[0])]
                elif len(param_vals)==1:
                    parameter_settings[parameter] =["fixed", float(param_vals[0])]
    return parameter_settings     

def exent_parameter_settings(ps):
    # add dummy "error" if not optimized
    for key in ps.keys():
        if len(ps[key]) == 2:
            ps[key].append(0)

    for rate in ["lambda", "q"]:
        std = np.sqrt(ps["var_"+rate][1]/(2.*ps["gamma_"+rate][1]))
        std_err2 = (1/2. * 1/np.sqrt( ps["var_"+rate][1]*ps["gamma_"+rate][1]) * ps["var_"+rate][2])**2 + \
                        (1/2. * np.sqrt(ps["var_"+rate][1]/2.) * ps["gamma_"+rate][1]**(-3/2.)* ps["gamma_"+rate][2])**2
        ps["std_"+rate] = ["derived", std, np.sqrt(std_err2)]

    for rate in ["lambda", "q"]:
        std = np.sqrt(ps["var_"+rate][1]/(2.*ps["gamma_"+rate][1]))/ps["mean_"+rate][1]
        std_err2 = (1/2. * 1/np.sqrt( ps["var_"+rate][1]*ps["gamma_"+rate][1])/ps["mean_"+rate][1] * ps["var_"+rate][2])**2 + \
                        (1/2. * np.sqrt(ps["var_"+rate][1]/2.) * ps["gamma_"+rate][1]**(-3/2.)/ps["mean_"+rate][1]* ps["gamma_"+rate][2])**2 + np.sqrt(ps["var_"+rate][1]/(2.*ps["gamma_"+rate][1]))/ps["mean_"+rate][1]**2 * ps["mean_"+rate][2]**2
        
        ps["cv_"+rate] = ["derived", std, np.sqrt(std_err2)]
        
    for xg in ["x", "g"]:
        sigma = np.sqrt(ps["var_"+xg][1])
        sigma_err = 1/np.sqrt(ps["var_"+xg][1]) * ps["var_"+xg][2]
        ps["sigma_"+xg] = ["derived", sigma, sigma_err]
        
    return ps

def read_header(filename, epsilon_err=0.05, read_state="final"):
    param_lines = header_lines(filename, until="10")+1
    parameters_arr = np.genfromtxt(filename, delimiter=',', dtype=str, max_rows=param_lines)
    param_dict = {}
    if read_state=="final":
        idx = 7
    elif read_state=="init":
        idx = 3
    for param in parameters_arr[1:]:
        if not is_emptystr(param[4]) and not is_emptystr(param[5]) and not is_emptystr(param[6]):
            param_dict[param[1]] = ["bound", float(param[idx])]
        elif not is_emptystr(param[4]):
            param_dict[param[1]] = ["free", float(param[idx])]
        else:
            param_dict[param[1]] = ["fixed", float(param[idx])]

    if (filename[:-4].endswith("final")):
        with open(filename,'r') as fin:
            for i, line in enumerate(fin):
                if line.startswith("epsilon"):
                    param_error_colums = line.strip().split(',')[1:]
                elif line.strip().split(',')[0] == str(epsilon_err):
                    param_error = np.sqrt(np.array(line.strip().split(',')[1:]).astype(float))
        for i, k in enumerate(param_error_colums):
            param_dict[k].append(param_error[i])
    return param_dict


def read_footer(filename):
    skip = True
    config = {}
    with open(filename,'r') as fin:
        for i, line in enumerate(fin):
            if line.startswith("n_data_points"):
                skip=False
            if not skip:
                config[line.split(',')[0].strip()] = line.split(',')[1].strip()
    return config

    
def get_suffix_files(directory, suffix='prediction.csv'):
    # given a directory returns all files that end with suffix
    entries = os.listdir(directory)
    final_files = []
    for e in entries:
        if e.endswith(suffix):
            final_files.append( os.path.join(directory, e) )
    return final_files




    
    
def legend_without_duplicate_labels(ax, sorting=None):
    def find_str(arr, s):
        for i,a in enumerate(arr):
            if a==s:
                return i
            
    handles, labels = ax.get_legend_handles_labels()
    if sorting=="auto":
        sorting = np.argsort(labels)
        handles = np.array(handles)[sorting]
        labels = np.array(labels)[sorting]
    elif sorting!=None:
        sorting = [s for s in sorting if s in labels]
        idx_sorting = []
        for s in sorting:
            idx_sorting.append(find_str(labels,s))
        handles = np.array(handles)[idx_sorting]
        labels = np.array(labels)[idx_sorting]

    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    return zip(*unique)

