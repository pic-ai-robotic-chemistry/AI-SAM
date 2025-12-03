
import numpy as np
import pandas as pd
from cuspy import ConfigUtils

from ai_sam.print_mols_utils import PrintMolsUtils

class SortProcess:
    def __init__(self, config):
        self.config = config

    def filter_by_homo(self):
        homo_lumo_df = pd.read_csv(self.config.homo_lumo_results_fp)
        smi_df = pd.read_csv(self.config.candidate_combine_fp, sep='\t')
        homo_lumo_df['mid'] = homo_lumo_df['Folder'].map(lambda x: int(x.split('_')[-1]))
        smis = smi_df['smiles'].tolist()
        mid_to_smiles = {i: smiles for i, smiles in enumerate(smis)}
        print(len(set(smis)))
        labels = []
        smis = []
        homos = []
        for _, row in homo_lumo_df.iterrows():
            homo = row['HOMO (eV)'][:4]
            lumo = row['LUMO (eV)'][:4]
            labels.append(f'HOMO({homo}), LUMO({lumo})')
            smis.append(mid_to_smiles[int(row['Folder'].split('_')[-1])])
            if homo.startswith('E'):
                homos.append(0)
            else:
                homos.append(float(homo))

        sorted_index = np.argsort(homos)
        sorted_labels = [labels[i] for i in sorted_index]
        print(len(set(smis)))
        sorted_smis = [smis[i] for i in sorted_index]
        print(len(set(sorted_smis)))
        sorted_mids = [list(mid_to_smiles.keys())[i] for i in sorted_index]

        PrintMolsUtils.print_mols(sorted_smis, sorted_mids, sorted_labels, pdf_fp=self.config.homo_lumo_mols_fp, temp_dp=self.config.temp_dp)


if __name__ == '__main__':
    conf = ConfigUtils.load_config('../config.json').solar_config
    cf = SortProcess(conf)
    cf.filter_by_homo()