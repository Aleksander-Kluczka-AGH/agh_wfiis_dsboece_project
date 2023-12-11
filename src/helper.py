import re
import os
import gzip
import pandas as pd

def read_gzip_file(file_path):
    with gzip.open(file_path) as f:
        data1 = pd.read_csv(f,sep=";",skiprows=[0,1],header=None)
    return data1

def read_all_data(path):
    dataFrames = []
    #iterate through all files
    for file in os.listdir(path):
        if file.endswith(".gzip"):
            file_path = f"{path}/{file}"
            # call read text file function
            dataFrames.append(read_gzip_file(file_path))
    data = pd.concat(dataFrames, ignore_index=True)
    data.columns=['time','latitude','longitude','nw data','accuracy']
    return data

def _get_param(val: str, name: str):
    m = re.match(f'^.* {name}=([^ ]+).*$', val)
    if m:
        return m.group(1)
    return None

def get_param(cell, name: str):
    if type(cell) is list:
        return [get_param(c, name) for c in cell]
    elif type(cell) is dict:
        param = _get_param(cell['signal'], name)
        if param:
            return param
        return _get_param(cell['identity'], name)
    return None

class CellInfo:
    def __init__(self, s: str):
        all_info = s.split('!')
        self.num_cells = int(all_info[0])
        sig_id = all_info[1:]
        self._info = [{'type':self.__get_type(sig_id[i+1]), 'signal':sig_id[i], 'identity':sig_id[i+1]}
                      for i in range(0, self.num_cells, 2)]

    def __str__(self):
        wcdma_len = len(self.wcdma())
        lte_len = len(self.lte())
        return f"CellInfo: n={self.num_cells} wcdma={wcdma_len} lte={lte_len}"

    def __repr__(self):
        return str(self)

    def cell_types(self):
        return set([cell['type'] for cell in self._info])

    def all(self):
        return self._info

    def wcdma(self):
        return [i for i in self._info if i['type'] == 'Wcdma']

    def lte(self):
        return [i for i in self._info if i['type'] == 'Lte']

    def get_by_pci(self, pci: str):
        return [cell for cell in self.lte()
                if get_param(cell, 'mPci') == pci]

    def get_by_mcc_mnc(self, mcc: str, mnc: str):
        return [cell for cell in self.all()
                if get_param(cell, 'mMcc') == mcc and get_param(cell, 'mMnc') == mnc]

    def get_attached(self):
        return [cell for cell in self.all() if get_param(cell, 'mMnc') != 'null']

    def get_attached_lte(self):
        return [cell for cell in self.lte() if get_param(cell, 'mMnc') != 'null']

    def get_attached_wcdma(self):
        return [cell for cell in self.wcdma() if get_param(cell, 'mMnc') != 'null']

    def __get_type(self, cell_id_str: str):
        return cell_id_str.split(':')[0][len('CellIdentity'):]

def explode_nw_data(df: pd.DataFrame, col_name='nw data'):
    exploded_df = df.copy()
    exploded_df[col_name] = df[col_name].apply(lambda txt: CellInfo(txt))
    return exploded_df

def rsrp_per_pci(df: pd.DataFrame, PCI: int, agg=min):
    pci_df = df[df['pci'].apply(lambda x: PCI in x)].copy()
    pci_df['rsrp'] = pci_df['nw data'].apply(
        lambda x: agg([float(get_param(pcell, 'rsrp')) for pcell in x.get_by_pci(str(PCI))]))
    return pci_df

def _at_least_one(values, search):
    for v in values:
        if v in search:
            return True
    return False

def _join(list_of_lists):
    joined_list = []
    for l in list_of_lists:
        joined_list += l
    return joined_list

def rsrp_per_multiple_pci(df: pd.DataFrame, PCIs: list, agg=min):
    pci_df = df[df['pci'].apply(_at_least_one, args=(PCIs,))].copy()
    pci_df['rsrp'] = pci_df['nw data'].apply(
        lambda x: agg([float(get_param(pcell, 'rsrp')) for pcell in _join([
            x.get_by_pci(str(pci)) for pci in PCIs]) ]))
    return pci_df

def rsrp_per_plmn(df: pd.DataFrame, mcc: str, mnc: str, agg=min):
    plmn_df = df[df['plmn lte'].apply(lambda x: '-'.join([mcc, mnc]) in x)].copy()
    plmn_df['rsrp'] = plmn_df['nw data'].apply(
        lambda x: agg([float(get_param(pcell, 'rsrp')) for pcell in x.get_by_mcc_mnc(mcc, mnc)]))
    return plmn_df

def rscp_per_plmn(df: pd.DataFrame, mcc: str, mnc: str, agg=min):
    plmn_df = df[df['plmn wcdma'].apply(lambda x: '-'.join([mcc, mnc]) in x)].copy()
    plmn_df['rscp'] = plmn_df['nw data'].apply(
        lambda x: agg([float(get_param(pcell, 'rscp')) for pcell in x.get_by_mcc_mnc(mcc, mnc) if get_param(pcell, 'rscp') is not None]))
    return plmn_df

def get_plmn(cell: dict):
    mcc = get_param(cell, 'mMcc')
    mnc = get_param(cell, 'mMnc')
    return '-'.join([mcc, mnc])

def expand_dataframe(df: pd.DataFrame):
    df['pci'] = df['nw data'].apply(lambda x: [int(p) for p in get_param(x.get_attached_lte(), 'mPci') if p is not None])
    df['plmn lte'] = df['nw data'].apply(lambda x: set([get_plmn(cell) for cell in x.get_attached_lte()]))
    df['plmn wcdma'] = df['nw data'].apply(lambda x: set([get_plmn(cell) for cell in x.get_attached_wcdma()]))
    return df