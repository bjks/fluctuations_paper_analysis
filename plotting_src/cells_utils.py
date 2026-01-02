import copy
import numpy as np
import os

def mk_missing_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory) 
    return directory


def find_cell(cells, cell_id):
    for cell in cells:
        if cell.cell_id == cell_id:
            return cell
    return None

                    
def fetch_cell_rank(cell_id, raw_data_pd, cell_id_col="sub_cell", cell_rank_col="cell_rank"):
    return raw_data_pd[raw_data_pd[cell_id_col]==cell_id][cell_rank_col].to_numpy()[0]
    

def assign_cell_rank_to_cells(cells, raw_data_pd, cell_id_col="sub_cell", cell_rank_col="cell_rank"):
    for i, cell in enumerate(cells):
        cells[i].cell_rank = fetch_cell_rank(cells[i].cell_id, raw_data_pd, cell_id_col, cell_rank_col)
    return cells


def get_mother_cells(cells):
    out_cells = []
    for i, _ in enumerate(cells):
        if cells[i].cell_rank == 0:
            out_cells.append(cells[i])
    return out_cells


def get_parent(cell, all_cells):
    for i, c in enumerate(all_cells):
        if c.cell_id == cell.parent_id:
            return i, c
    return None, None


def get_roots(all_cells):
    roots = []
    for cell in all_cells:
        i, c = get_parent(cell, all_cells)
        if i==None:
            roots.append(cell)
    return roots
    
    
def get_daughters(cell, all_cells):
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


def get_path_idx(cells, idx0=0):
    idx = [idx0]
    last_cell = cells[idx0].cell_id
    for i, _ in enumerate(cells):
        if last_cell == cells[i].parent_id: 
            last_cell = cells[i].cell_id
            idx.append(i)
    return idx


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


def calc_concentration(cells):
    for i, _ in enumerate(cells):
        cells[i].concentration = cells[i].gfp/np.exp(cells[i].log_length)
    return cells

def get_conc_from_cells(cells):
    conc = np.array([])
    for cell in cells:
        conc = np.append(conc, cell.concentration)
    return conc

def get_log_length_from_cells(cells):
    ll = np.array([])
    for cell in cells:
        ll = np.append(ll, cell.log_length)
    return ll

def get_length_from_cells(cells):
    ll = np.array([])
    for cell in cells:
        ll = np.append(ll, np.exp(cell.log_length))
    return ll

def get_gfp_from_cells(cells):
    gfp = np.array([])
    for cell in cells:
        gfp = np.append(gfp, cell.gfp)
    return gfp