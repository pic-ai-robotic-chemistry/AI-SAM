
from rdkit.Chem import AllChem
from tqdm import tqdm
import pandas as pd
from cuspy import ConfigUtils

from solar_designer.mol_utils import MolUtils


class ExtractGroup:

    def __init__(self, config):
        self.config = config
        self.carbazole = AllChem.MolFromSmiles('c12ccccc1c3c(cccc3)[NH]2')

    def get_group(self, smiles: str):
        mol = AllChem.MolFromSmiles(smiles)
        carbazole_part = mol.GetSubstructMatch(self.carbazole)
        group_idxes = []
        for idx in range(mol.GetNumAtoms()):
            if idx not in carbazole_part:
                group_idxes.append(idx)

        frag, old_idx_to_new, link_idxes = MolUtils.frag_idxes_to_mol_with_link(mol, group_idxes)
        return AllChem.MolToSmiles(frag), link_idxes

    def extract_group(self):
        query_mol_df = pd.read_csv(self.config.query_again_mol_fp, encoding='utf-8')
        group_smarts_to_count = {}
        all_link_idxes = []
        for _, row in tqdm(query_mol_df.iterrows(), total=len(query_mol_df)):
            mid = row['mid']
            smiles = row['smiles']
            group_smarts_merge, link_idxes = self.get_group(smiles)
            for group_smarts in group_smarts_merge.split('.'):
                if group_smarts not in group_smarts_to_count:
                    group_smarts_to_count[group_smarts] = 1
                    # all_link_idxes.append(link_idxes)
                else:
                    group_smarts_to_count[group_smarts] += 1
        group_smarts_to_count_df = pd.DataFrame({'group_smiles': list(group_smarts_to_count.keys()),
                                                 'num': list(group_smarts_to_count.values())})
        group_smarts_to_count_df = group_smarts_to_count_df.sort_values(by='num', ascending=False)
        group_smarts_to_count_df.to_csv(self.config.group_counts_fp, sep='\t', index=False)


if __name__ == '__main__':
    conf = ConfigUtils.load_config('../config.json').solar_config
    eg = ExtractGroup(conf)
    eg.extract_group()