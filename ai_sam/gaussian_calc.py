import os

import pandas as pd
from cuspy import ConfigUtils
from tqdm import tqdm

from mol_utils import MolUtils


class GaussianCalc:

    def __init__(self, config):
        self.config = config

    def load_mids(self):
        homo_lumo_df = pd.read_csv(self.config.homo_lumo_results_fp)
        homo_lumo_df = homo_lumo_df.loc[homo_lumo_df['HOMO (eV)'].map(lambda x: not x.startswith('Err'))]
        homo_lumo_df['HOMO (eV)'] = homo_lumo_df['HOMO (eV)'].map(float)
        homo_lumo_df = homo_lumo_df.loc[homo_lumo_df['HOMO (eV)'].map(lambda x: x > -5.8)]
        homo_lumo_df = homo_lumo_df.loc[homo_lumo_df['HOMO (eV)'].map(lambda x: x < -5.4)]
        print(f"num of mols: {len(homo_lumo_df)}")
        return [int(fp.split('_')[-1]) for fp in homo_lumo_df['Folder'].tolist()]

    def prepare_init_files(self):
        mol_df = pd.read_csv(self.config.candidate_combine_fp, sep='\t')
        for n, (_, row) in enumerate(mol_df.iterrows()):
            smiles = row['smiles']
            dp = f'../data_calc/mol_{n}'
            os.mkdir(dp)
            gjf_fp = os.path.join(dp, f'mol_{n}.gjf')
            MolUtils.smiles_to_gjf_file(smiles, str(n), gjf_fp)

    def prepare_cation_init_file(self):
        mids = self.load_mids()
        mol_df = pd.read_csv(self.config.candidate_combine_fp, sep='\t')
        for n, (_, row) in enumerate(mol_df.iterrows()):
            if n not in mids:
                continue
            smiles = row['smiles']
            dp = f'../data_calc_cation/mol_{n}'
            if os.path.exists(dp):
                continue
            os.mkdir(dp)
            gjf_fp = os.path.join(dp, f'mol_{n}.gjf')
            MolUtils.smiles_to_gjf_file(smiles, str(n), gjf_fp, charge=1, spin=2)

    def prepare_neural_to_cation_init_file(self):
        mids = self.load_mids()
        os.mkdir('../data_calc_neural_to_cation')
        for mid in mids:
            fchk_fp = f'../data_calc/mol_{mid}/out_{mid}.fchk'
            if os.path.exists(fchk_fp):
                gjf_dp = f'../data_calc_neural_to_cation/mol_{mid}'
                if os.path.exists(gjf_dp):
                    continue
                os.mkdir(gjf_dp)
                gjf_fp = os.path.join(gjf_dp, f'mol_{mid}.gjf')
                MolUtils.fchk_to_gjf(fchk_fp, gjf_fp, mid)

    def prepare_cation_to_neural_init_file(self):
        mids = self.load_mids()
        os.mkdir('../data_calc_cation_to_neural')
        for mid in mids:
            fchk_fp = f'../data_calc_cation/mol_{mid}/out_{mid}.fchk'
            if os.path.exists(fchk_fp):
                gjf_dp = f'../data_calc_cation_to_neural/mol_{mid}'
                if os.path.exists(gjf_dp):
                    continue
                os.mkdir(gjf_dp)
                gjf_fp = os.path.join(gjf_dp, f'mol_{mid}.gjf')
                MolUtils.fchk_to_gjf(fchk_fp, gjf_fp, mid, charge=0, multiplicity=1)

    def prepare_init_tzvp_files(self):
        for n in range(300):
            dp = f'../data_calc/mol_{n}'
            fchk_fp = os.path.join(dp, f'out_{n}.fchk')
            gjf_fp = os.path.join(dp, f'mol_tzvp_{n}.gjf')
            MolUtils.fchk_to_gjf(fchk_fp, gjf_fp, n)

    def fchk_to_energy(self, fchk_fp: str):
        energy = None
        try:
            with open(fchk_fp, 'r') as file:
                for line in file:
                    # 查找包含能量信息的行
                    if 'Total Energy' in line:
                        parts = line.split()
                        # 假设能量在行的最后一个部分
                        energy = float(parts[-1]) * 27.2114
                        break
        except FileNotFoundError:
            print(f"文件 {fchk_fp} 未找到。")
        except Exception as e:
            print(f"处理文件时出错: {e}")

        return energy

    def extract_dipole_moment_from_fchk(self, fchk_file):
        dipole_moment = []
        with open(fchk_file, 'r') as file:
            for line in file:
                # 找到偶极矩的起始行
                if "Dipole Moment" in line:
                    # 偶极矩分量通常在这一行后面
                    dipole_moment = list(map(float, next(file).split()))
                    break

        if len(dipole_moment) == 3:
            dx, dy, dz = dipole_moment
            total_dipole = (dx**2 + dy**2 + dz**2)**0.5
            return dx, dy, dz, total_dipole
        else:
            print(f"未能在文件({fchk_file})中找到完整的偶极矩信息: {dipole_moment}")
            return None, None, None, None


    def extract_hole_re(self):
        mids = self.load_mids()
        mol_df = pd.read_csv(self.config.candidate_combine_fp, sep='\t')
        smis = mol_df['smiles'].tolist()
        log_mids = []
        log_smis = []
        log_neural = []
        log_cation = []
        log_neural_cation = []
        log_cation_neural = []
        for mid in mids:
            neural_fchk = f'../data_calc/mol_{mid}/out_{mid}.fchk'
            cation_fchk = f'../data_calc_cation/mol_{mid}/out_{mid}.fchk'
            neural_to_cation_fchk = f'../data_calc_neural_to_cation/mol_{mid}/out_{mid}.fchk'
            cation_to_neural_fchk = f'../data_calc_cation_to_neural/mol_{mid}/out_{mid}.fchk'
            log_smis.append(smis[mid])
            log_neural.append(self.fchk_to_energy(neural_fchk))
            log_cation.append(self.fchk_to_energy(cation_fchk))
            log_neural_cation.append(self.fchk_to_energy(neural_to_cation_fchk))
            log_cation_neural.append(self.fchk_to_energy(cation_to_neural_fchk))
        df = pd.DataFrame({'mid': mids, 'smiles': log_smis, 'neural': log_neural, 'cation': log_cation,
                           'neural_cation': log_neural_cation, 'cation_neural': log_cation_neural})
        df['HRE (eV)'] = df['cation_neural'] - df['neural'] + df['neural_cation'] - df['cation']
        df['IE (eV)'] = df['cation'] - df['neural']
        df.to_csv(self.config.hole_reorgnization_fp, index=False)

    def extract_dipole_moment(self):
        res_dict = {'mid': [], 'x': [], 'y': [], 'z': [], 'tot': []}
        for mid in tqdm(range(300), total=300):
            fchk_fp = f'../data_calc/mol_{mid}/out_{mid}.fchk'
            x, y, z, tot = self.extract_dipole_moment_from_fchk(fchk_fp)
            res_dict['mid'].append(mid)
            res_dict['x'].append(x)
            res_dict['y'].append(y)
            res_dict['z'].append(z)
            res_dict['tot'].append(tot)
        res_df = pd.DataFrame(res_dict)
        res_df.to_csv(self.config.dipole_moment_fp, index=False)


if __name__ == '__main__':
    pass
