import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import subprocess
import os


SMALL_SIZE = 13
MEDIUM_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


# ========================================== #
# =============== CELL CLASS =============== #
# ========================================== #
class Cell:
    def __init__(self, log_length0, gfp0, lambda0, q0, time0=0., cell_id = 0, parent_id=-1):
        self.parent_id = parent_id
        self.cell_id = cell_id
        self.length = [np.exp(log_length0)]  # s(t)
        self.log_length = [log_length0]      # x(t) = x0 + int lambda dt
        self.gfp = [gfp0]
        self.lt = [lambda0]
        self.qt = [q0]
        self.time = [time0]
        self.segment = []

    def to_df(self, n=1, start=0):
        df = pd.DataFrame({   "cell_id": ([self.cell_id]*len(self.time))[start::n],
                                "time_min": self.time[start::n],
                                "parent_id": ([self.parent_id]*len(self.time))[start::n],
                                "log_length": self.log_length[start::n], 
                                "gfp": self.gfp[start::n],
                                "lt": self.lt[start::n],
                                "qt": self.qt[start::n]})
        if len(self.segment)>0:
            df['segment']=self.segment[start::n]
        return df


def ggp_df2cells(dataset, time="time", 
            log_length="mean_x", gfp="mean_g", 
            lt="mean_l", qt="mean_q",
            cov_xx="cov_xx",
            cov_gg="cov_gg",
            cov_ll="cov_ll",
            cov_qq="cov_qq",
            cell_id="cell_id", 
            parent_id="parent_id", 
            lane=None):
    """ 
    dataset (pandas data frame as read from csv file) to list of Cell instances, m
    written for ggp output
    """
    cell_list = []
    last_cell = ""
    for _, row in dataset.iterrows(): 
        if row[cell_id] != last_cell:
            c = str(row[cell_id])
            p = str(row[parent_id])

            lambda0 = row[lt]
            q0 = row[qt]

            new_cell = Cell(row[log_length], row[gfp], 
                        lambda0, q0, 
                        time0=row[time],
                        cell_id=c, 
                        parent_id=p)
            cell_list.append(new_cell)
            cell_list[-1].cov_xx = []
            cell_list[-1].cov_gg = []
            cell_list[-1].cov_ll = []
            cell_list[-1].cov_qq = []

        else:
            cell_list[-1].log_length.append(row[log_length])
            cell_list[-1].gfp.append(row[gfp])
            cell_list[-1].time.append(row[time])

            cell_list[-1].lt.append(row[lt])
            cell_list[-1].qt.append(row[qt])

        cell_list[-1].cov_xx.append(row[cov_xx])
        cell_list[-1].cov_gg.append(row[cov_gg])
        cell_list[-1].cov_ll.append(row[cov_ll])
        cell_list[-1].cov_qq.append(row[cov_qq])

        last_cell = row[cell_id]
    return cell_list


# ========================================== #
# =============== READING    =============== #
# ========================================== #
def header_lines(filename, until="cell_id"):
    with open(filename,'r') as fin:
        for i, line in enumerate(fin):
            if line.startswith(until):
                return i
    return None

def is_emptystr(s):
    return len(s.strip())==0

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


# ========================================== #
# =============== HELPERS    =============== #
# ========================================== #

def get_parent(cell, all_cells):
    for i, c in enumerate(all_cells):
        if c.cell_id == cell.parent_id:
            return i, c
    return None, None

def get_path_from_leaf(cells, idx0=0, rev=True):
    idx = []
    last_cell = cells[idx0]
    i = idx0
    while last_cell != None:
        idx.append(i)        
        i, last_cell = get_parent(last_cell, cells)
    if rev:
        return idx[::-1]
    return idx


def get_leafs(cells):
    idx = []
    parent_ids = [cells[i].parent_id for i, _ in enumerate(cells)]
    for i, _ in enumerate(cells):
        if cells[i].cell_id not in parent_ids:
            idx.append(i)
    return idx


def get_longest_path(cells):
    idx = get_leafs(cells)
    longest_path = []
    for i in idx: 
#         path = get_path_idx(cells, i)
        path = get_path_from_leaf(cells, i)
        if len(path) > len(longest_path):
            longest_path = path
    return longest_path



def get_cell_by_id(cell_id, cells):
    for cell in cells:
        if cell.cell_id == cell_id:
            return cell

def get_roots(cells):
    idx = []
    cell_ids = [cells[i].cell_id for i, _ in enumerate(cells)]
    for i, _ in enumerate(cells):
        if cells[i].parent_id not in cell_ids:
            idx.append(i)
    return idx

def get_daughters(cell, all_cells):
    """ 
    Get the daugter cells
  
    Parameters:
    cell (Cell): current cell
    all_cells (list of Cell): all cells
    
    Returns:
    list of daughter cells, 2 long, entries are None if there is no daugher cell(s)
    """
    ds = [None, None]
    for c in all_cells:
        if c.parent_id == cell.cell_id:
            if ds[0]==None:
                ds[0] = c
            elif ds[1] == None:
                ds[1] = c
            else:
                print("More than 2 daughters")
    return ds

def get_daughters_idx(cell, all_cells):
    """ Same as get_daughters but returns indices
    """
    ds = [None, None]
    for i, c in enumerate(all_cells):
        if c.parent_id == cell.cell_id:
            if ds[0]==None:
                ds[0] = i
            elif ds[1] == None:
                ds[1] = i
            else:
                print("More than 2 daughters")
    return ds


def get_lambda(cell, t):
    return cell.lt[t-1]

def get_q(cell, t):
    return cell.qt[t-1]



### PLOTTING ###
    
def plot_concentration(cells_integrated, cells, path, plot_file=None, cells_raw=None, label_i=None):
    fig = plt.figure(figsize=(8.3,4.8))
        
    ax = plt.axes()
    
    ax.set_xlabel("time (min)")
    ax.set_ylabel("concentration")

    for a in [ax]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

        a.spines['right'].set_color('none')
        a.yaxis.tick_left()

        a.spines['top'].set_color('none')
        a.xaxis.tick_bottom()
        
    label_p = "prediction"
    label_i = label_i
    label_r = "raw"
    
    for j, i in enumerate(path):
        plt.plot(cells_integrated[i].time, np.array(cells_integrated[i].gfp)/np.exp(cells_integrated[i].log_length), 
                 color="tab:green", label=label_i)
        plt.plot(cells[i].time, np.array(cells[i].gfp)/np.exp(cells[i].log_length), 
                 color="tab:grey", label=label_p)    
        if cells_raw!=None:
            plt.plot(cells_raw[i].time, np.array(cells_raw[i].gfp)/np.exp(cells_raw[i].log_length), 
                     '-o', markersize=2, lw=0.5, 
                     color="tab:grey", label=label_r,alpha=0.3)    
        label_p = None
        label_i = None
        label_r = None
    ax.legend(loc='upper left', ncol=1)
    if plot_file!=None:
        plt.savefig(plot_file, bbox_inches='tight', facecolor="white")
        plt.close()
    else:
        plt.show()
    
        
def plot_concentration_hist(cells_integrated, cells, plot_file=None, label_i=None, log=True):
    
    conc_real = np.array([])
    conc_int = np.array([])
    for i, _ in enumerate(cells_integrated):
        conc_real = np.append(conc_real, 
                              np.array(cells[i].gfp)/np.exp(cells[i].log_length))
        conc_int = np.append(conc_int, 
                             np.array(cells_integrated[i].gfp)/np.exp(cells_integrated[i].log_length))
        
    fig = plt.figure(figsize=(8.3,4.8))
    ax = plt.axes()
    
    ax.set_xlabel("concentration")
    ax.set_ylabel("frequency")

    for a in [ax]:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

        a.spines['right'].set_color('none')
        a.yaxis.tick_left()

        a.spines['top'].set_color('none')
        a.xaxis.tick_bottom()
        
    def estimate_iqr(c):
        return np.subtract(*np.percentile(c, [75, 25]))

    def get_label(label, dat):
        return label + r', $\mu$={:.2E} $\sigma=${:.2E}'.format(np.mean(dat), estimate_iqr(dat))
    
    _, bin_edges_both = np.histogram(np.log(np.append(conc_real,conc_int )), density=False, bins=100)
    hist, bin_edges = np.histogram(np.log(conc_real), density=False, bins=bin_edges_both)
#     hist= np.array(hist)/(np.sum(hist))
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
            edgecolor=None, align="edge", color="tab:grey", alpha=0.5, label=get_label("data",np.log(conc_real)))
    
    hist, bin_edges = np.histogram(np.log(conc_int), density=False, bins=bin_edges_both)
#     hist= np.array(hist)/(np.sum(hist))
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges),
            edgecolor=None, align="edge", color="tab:green", alpha=0.5, label=get_label(label_i,np.log(conc_int)))
    ax.legend(loc='lower left', ncol=1, bbox_to_anchor=(0,1))
    if log:
        ax.set_yscale('log')

    if plot_file!=None:
        plt.savefig(plot_file, bbox_inches='tight', facecolor="white")
        plt.close()
    else:
        plt.show()

#plotting
def plot_x_g(cells_integrated, cells, path, plot_file=None, cells_raw=None, label_i=None):
    
    fig, axs = plt.subplots(2, 1, figsize=(8.3,4.8), sharex=True)
    
    ax = axs.ravel()
    ax[1].set_xlabel("time (min)")
    ax[0].set_ylabel("log length")
    ax[1].set_ylabel("gfp")

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

        a.spines['right'].set_color('none')
        a.yaxis.tick_left()

        a.spines['top'].set_color('none')
        a.xaxis.tick_bottom()
        
    label_p = "prediction"
    label_i = label_i
    label_r = "raw"
    
    for j, i in enumerate(path):
        ax[0].plot(cells_integrated[i].time, cells_integrated[i].log_length, 
                 color="tab:blue", label=label_i, alpha=0.5)
        ax[0].plot(cells[i].time, cells[i].log_length, 
                 color="tab:grey", label=label_p, alpha=0.5)    
                   
        ax[1].plot(cells_integrated[i].time, cells_integrated[i].gfp, 
                 color="tab:orange", label=label_i, alpha=0.5)
        ax[1].plot(cells[i].time, cells[i].gfp, 
                 color="tab:grey", label=label_p, alpha=0.5) 
                   
        if cells_raw!=None:
            ax[0].plot(cells_raw[i].time, cells_raw[i].log_length, 
                     '-o', markersize=2, lw=0.5, 
                     color="tab:grey", label=label_r, alpha=0.3)    
            ax[1].plot(cells_raw[i].time, cells_raw[i].gfp, 
                     '-o', markersize=2, lw=0.5, 
                     color="tab:grey", label=label_r, alpha=0.3)  
        
        label_p = None
        label_i = None
        label_r = None
    ax[0].legend(loc='upper left', ncol=1)
    if plot_file!=None:
        plt.savefig(plot_file, bbox_inches='tight', facecolor="white")
        plt.close()
    else:
        plt.show()    
        

### Save cells (as csv)
        
def save_cells(cells, outfile):
    dataset = pd.DataFrame()
    for i in range(len(cells)):
        next_celldf = cells[i].to_df(1)
        dataset = dataset.append(next_celldf)
    dataset.to_csv(outfile)
    
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