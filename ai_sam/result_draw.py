import math

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from cuspy import ConfigUtils
from rdkit.Chem import AllChem, Descriptors
import matplotlib.gridspec as gridspec


class Draw:

    def __init__(self, config):
        self.config = config

    def draw_disturb(self, xs, bins, name, fp):

        plt.figure(figsize=(5, 4), dpi=300)
        n, bins, patches = plt.hist(xs, bins=bins, density=True, alpha=0.8,
                                    edgecolor='white', label=f'{name} Distribution')
        loc, scale = stats.cauchy.fit(xs)
        x = np.linspace(min(xs), max(xs), 1000)
        cauchy_pdf = stats.cauchy.pdf(x, loc=loc, scale=scale)
        plt.plot(x, cauchy_pdf, '-', lw=2, label='Cauchy Distribution Fit')
        plt.xlabel(f"{name} Values (eV)")
        plt.ylabel("Frequency Density")
        plt.title(f"{name} Values Distribution")
        plt.legend()
        plt.savefig(fp)
        plt.clf()

    def draw_radia(self):

        # 数据
        labels = ['A', 'B', 'C', 'D']  # 四个维度
        values = [4, 3, 2, 5]  # 每个维度的值
        values += values[:1]  # 闭合数据环（首尾相连）

        # 计算角度
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合角度环

        # 设置画布
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # 绘制雷达图
        ax.fill(angles, values, color='blue', alpha=0.25)  # 填充区域
        ax.plot(angles, values, color='blue', linewidth=2)  # 边界线

        # 添加标签
        ax.set_yticks([1, 2, 3, 4, 5])  # 设置径向刻度
        ax.set_yticklabels(['1', '2', '3', '4', '5'])
        ax.set_xticks(angles[:-1])  # 设置角度刻度
        ax.set_xticklabels(labels)

        # 显示图形
        plt.title('四角形雷达图', size=16, pad=20)
        plt.show()

    def draw_homo_lumo_distribute(self):
        homo_lumo_df = pd.read_csv(self.config.homo_lumo_results_fp)
        homo_lumo_df = homo_lumo_df.loc[homo_lumo_df['HOMO (eV)'].map(lambda x: x != 'Error')]
        homo_lumo_df['HOMO (eV)'] = homo_lumo_df['HOMO (eV)'].map(float)
        homo_lumo_df['LUMO (eV)'] = homo_lumo_df['LUMO (eV)'].map(float)
        print(f"num of homo: {len(homo_lumo_df)}")
        homos = homo_lumo_df['HOMO (eV)'].tolist()
        lumos = homo_lumo_df['LUMO (eV)'].tolist()
        self.draw_disturb(homos, 30, 'HOMO', self.config.homo_disturb_fig_fp)
        self.draw_disturb(lumos, 30, 'LUMO', self.config.lumo_disturb_fig_fp)

    def draw_her_distribute(self):
        her_df = pd.read_csv(self.config.hole_reorgnization_fp)
        hers = her_df['HRE (eV)'].tolist()
        self.draw_disturb(hers, 30, '$\lambda_{h}$', self.config.hre_distribute_fig_fp)

    def draw_retrosyn_distribute(self):
        retro_df = pd.read_csv(self.config.retrosyn_result_fp, sep='\t')
        scores = retro_df['score'].tolist()
        self.draw_disturb(scores, 30, 'Synthesizability', self.config.retrosyn_distribute_fig_fp)

    def get_mid_to_new_mid(self):
        df = pd.read_csv(self.config.candidate_combine_fp, sep='\t')
        mids = df['mid'].tolist()
        mid_to_new_mid = {m: i for i, m in enumerate(mids)}
        return mid_to_new_mid

    def normalize(self, arr, positive=False):
        arr = np.array(arr)
        normalized_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        if positive:
            normalized_arr = 1 - normalized_arr
        return normalized_arr, arr.max(), arr.min()

    def num_aromatic(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        num_aro = 0
        for atom in mol.GetAtoms():
            if atom.GetIsAromatic():
                num_aro += 1
        return num_aro - 6

    def num_heteroatom(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        for symbol in ['F', 'S', 'Br', 'I', 'Cl']:
            if symbol in atom_symbols:
                return 1
        if atom_symbols.count('O') > 3:
            return 1
        if atom_symbols.count('N') > 1:
            return 1
        return 0.5

    def log_p(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        logp = Descriptors.MolLogP(mol)
        return logp

    def tpsa(self, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        tpsa = Descriptors.TPSA(mol)
        return tpsa

    def draw_all(self, hist_data, radia_data, border_data, labels):
        fig = plt.figure(figsize=(12, 12), dpi=300)
        gs = gridspec.GridSpec(4, 4, figure=fig)
        for n, label in enumerate(labels):
            print(f"{n} - {label}")
            ax = fig.add_subplot(gs[n // 4, n % 4])
            values = hist_data[label]
            if label == 'Heterosatom':
                n, bins, patches = ax.hist(values, bins=20, density=True, alpha=0.8,
                                           edgecolor='white', label=f'{label}')
            else:
                n, bins, patches = ax.hist(values, bins=20, density=True, alpha=0.8,
                                           edgecolor='white', label=f'{label}')
                loc, scale = stats.cauchy.fit(values)
                x = np.linspace(min(values), max(values), 1000)
                cauchy_pdf = stats.cauchy.pdf(x, loc=loc, scale=scale)
                ax.plot(x, cauchy_pdf, '-', lw=2, label='Cauchy Distribution Fit')
            # ax.set_yticks([])
            ax.set_xlabel(label)
            ax.set_ylabel('Frequency Density')

        ax_radia = fig.add_subplot(gs[2:4, 0:2], polar=True)
        good_mids = [2, 4, 79, 211]
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合角度环

        for i, mid in enumerate(radia_data['mid']):
            values = [radia_data[key][i] for key in labels]
            values.append(values[0])
            if mid in good_mids:
                ax_radia.plot(angles, values, linewidth=2, zorder=100)
            else:
                ax_radia.plot(angles, values, color='lightgray', alpha=1, linewidth=1, zorder=10)  # 边界线
        ax_radia.plot(angles, [0] * (num_vars + 1), color='black', linewidth=1, linestyle='-', zorder=100)
        ax_radia.plot(angles, [1] * (num_vars + 1), color='black', linewidth=1, linestyle='-', zorder=100)
        for i in range(num_vars):
            ax_radia.plot([angles[i], angles[i]], [0, 1], color='black', linewidth=1, zorder=50)

        for i, key in enumerate(labels):
            value = radia_data[key][i]
            borders = border_data[key]
            angle = angles[i] * (180 / math.pi)
            # if angle > 180:
            # angle -= 180
            angle_label = angle - 90
            if angle > 180:
                angle_label = angle_label - 180
            print(f"{key}: {angle}")
            ax_radia.text(angles[i], -0.1, str(round(borders[0], 2)), ha='center', va='center', rotation=angle)
            ax_radia.text(angles[i], 1.1, str(round(borders[1], 2)), ha='center', va='center', rotation=angle)
            ax_radia.text(angles[i], 1.25, key, ha='center', va='center', rotation=angle_label)

        for nr, wr in zip([0, 0.5], [0.25, 0.75]):
            # 绘制同心圆之间的填充颜色
            theta = np.linspace(0, 2 * np.pi, 100)  # 角度范围
            r_inner = nr  # 内圆半径
            r_outer = wr  # 外圆半径

            # 填充两个同心圆之间的区域
            ax_radia.fill(
                np.concatenate([theta, theta[::-1]]),  # 角度范围
                np.concatenate([np.full_like(theta, r_outer), np.full_like(theta, r_inner)]),  # 半径范围
                color="#DCEAF7", alpha=1  # 填充颜色和透明度
            )
        ax_radia.set_ylim(-0.4, 1)
        ax_radia.set_yticklabels([])
        ax_radia.set_xticks(angles[:-1])  # 设置角度刻度
        ax_radia.set_xticklabels([])
        ax_radia.tick_params(pad=20)
        ax_radia.grid(False)

        plt.tight_layout()
        # plt.show()
        plt.savefig('final_fig_without_synthetic.jpg')

    def draw_radia(self, radia_data, border_data, labels):
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合角度环

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

        good_mids = [2, 4, 79]

        for i, mid in enumerate(radia_data['mid']):
            values = [radia_data[key][i] for key in labels]
            values.append(values[0])
            if mid in good_mids:
                ax.plot(angles, values, linewidth=2, zorder=100)
            else:
                ax.plot(angles, values, color='lightgray', alpha=1, linewidth=1, zorder=10)  # 边界线
        ax.plot(angles, [0] * (num_vars + 1), color='black', linewidth=1, linestyle='-', zorder=100)
        ax.plot(angles, [1] * (num_vars + 1), color='black', linewidth=1, linestyle='-', zorder=100)
        for i in range(num_vars):
            ax.plot([angles[i], angles[i]], [0, 1], color='black', linewidth=1, zorder=50)

        for i, key in enumerate(labels):
            value = radia_data[key][i]
            borders = border_data[key]
            angle = angles[i] * (180 / math.pi)
            # if angle > 180:
            # angle -= 180
            angle_label = angle - 90
            if angle > 180:
                angle_label = angle_label - 180
            print(f"{key}: {angle}")
            ax.text(angles[i], -0.09, str(round(borders[0], 2)), ha='center', va='center', rotation=angle)
            ax.text(angles[i], 1.09, str(round(borders[1], 2)), ha='center', va='center', rotation=angle)
            ax.text(angles[i], 1.25, key, ha='center', va='center', rotation=angle_label)

        for nr, wr in zip([0, 0.5], [0.25, 0.75]):
            # 绘制同心圆之间的填充颜色
            theta = np.linspace(0, 2 * np.pi, 100)  # 角度范围
            r_inner = nr  # 内圆半径
            r_outer = wr  # 外圆半径

            # 填充两个同心圆之间的区域
            ax.fill(
                np.concatenate([theta, theta[::-1]]),  # 角度范围
                np.concatenate([np.full_like(theta, r_outer), np.full_like(theta, r_inner)]),  # 半径范围
                color="#DCEAF7", alpha=1  # 填充颜色和透明度
            )
        ax.set_ylim(-0.4, 1)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])  # 设置角度刻度
        ax.set_xticklabels([])
        ax.tick_params(pad=20)
        ax.grid(False)
        plt.tight_layout()
        plt.savefig(self.config.radia_fig_fp)

    def draw_final_score(self):
        homo_lumo_df = pd.read_csv(self.config.homo_lumo_results_fp)
        homo_lumo_df = homo_lumo_df.loc[homo_lumo_df['HOMO (eV)'].map(lambda x: x != 'Error')]
        homo_lumo_df['HOMO (eV)'] = homo_lumo_df['HOMO (eV)'].map(float)
        homo_lumo_df['LUMO (eV)'] = homo_lumo_df['LUMO (eV)'].map(float)
        homo_lumo_df['mid'] = homo_lumo_df['Folder'].map(lambda x: int(x.split('_')[-1]))

        her_df = pd.read_csv(self.config.hole_reorgnization_fp)
        her_df['IE (eV)'] = her_df['cation'] - her_df['neural']

        retro_df = pd.read_csv(self.config.retrosyn_result_fp, sep='\t')
        mid_to_new_mid = self.get_mid_to_new_mid()
        retro_df['mid'] = retro_df['mid'].map(lambda x: mid_to_new_mid[x])

        mids = her_df['mid'].tolist()

        new_mids = []
        for mid in mids:
            homo = homo_lumo_df.loc[homo_lumo_df['mid'] == mid].reset_index()['HOMO (eV)'][0]
            if round(homo, 2) >= -5.76:
                new_mids.append(mid)
        mids = new_mids

        dipole_moment_df = pd.read_csv(self.config.dipole_moment_fp)
        dipole_moment_df = dipole_moment_df.loc[dipole_moment_df['mid'].isin(mids)]

        homo_lumo_df = homo_lumo_df.loc[homo_lumo_df['mid'].isin(mids)]
        retro_df = retro_df.loc[retro_df['mid'].isin(mids)]

        print(f"len homo lumo: {len(homo_lumo_df)}; her: {len(her_df)}, retro: {len(retro_df)}")

        homos = []
        lumos = []
        hres = []
        retros = []
        dipoles = []
        aros = []
        heteros = []
        ies = []
        smis = []
        logps = []
        tpsas = []
        for mid in mids:
            smiles = her_df.loc[her_df['mid'] == mid].reset_index()['smiles'][0]
            smis.append(smiles)
            hl_row = homo_lumo_df.loc[homo_lumo_df['mid'] == mid].reset_index()
            homos.append(hl_row['HOMO (eV)'][0])
            lumos.append(hl_row['LUMO (eV)'][0])
            hres.append(her_df.loc[her_df['mid'] == mid].reset_index()['HRE (eV)'][0])
            retro = retro_df.loc[retro_df['mid'] == mid].reset_index()['score'][0]
            if mid in [2, 4, 79]:
                retro = 1
            if retro == 1:
                retro = 2
            retros.append(retro)
            dipoles.append(dipole_moment_df.loc[dipole_moment_df['mid'] == mid].reset_index()['tot'][0])
            aros.append(self.num_aromatic(smiles))
            heteros.append(self.num_heteroatom(smiles))
            ies.append(her_df.loc[her_df['mid'] == mid].reset_index()['IE (eV)'][0])
            logps.append(self.log_p(smiles))
            tpsas.append(self.tpsa(smiles))
            if mid in [79, 211, 2, 214, 4]:
                print('----')
                print(
                    f'mid: {mid}\nhomo: {round(homos[-1], 2)}\nlumos: {round(lumos[-1], 2)}\nlambda: {round(hres[-1], 2)}\nsynthesis: 2\ndipole: {round(dipoles[-1], 2)}\naromatics: {aros[-1]}\nheteroatoms: {heteros[-1]}\nie: {round(ies[-1], 2)}')
                print('----')
        gaps = [abs(h - l) for h, l in zip(homos, lumos)]

        hist_data = {
            'mid': mids,
            'smiles': smis,
            'HOMO (eV)': homos,
            'LUMO (eV)': lumos,
            'HOMO LUMO Gap (eV)': gaps,
            'IE (eV)': ies,
            'Heterosatom': heteros,
            'Dipole Moment': dipoles,
            'Aromatic': aros,
            'Synthesizability': retros,
            '$\lambda_{h}$ (eV)': hres,
            'LogP': logps,
            'TPSA': tpsas
        }

        homos, homo_max, homo_min = self.normalize(homos, positive=True)
        lumos, lumo_max, lumo_min = self.normalize(lumos)
        hres, hre_max, hre_min = self.normalize(hres, positive=True)
        reset_retros, retro_max, retro_min = self.normalize(retros)
        dipoles, dipole_max, dipole_min = self.normalize(dipoles)
        gaps, gap_max, gap_min = self.normalize(gaps)
        aros, aro_max, aro_min = self.normalize(aros)
        logps, logp_max, logp_min = self.normalize(logps, positive=True)
        tpsas, tpsa_max, tpsa_min = self.normalize(tpsas)

        ies = [abs(abs(ie) - abs(5.7)) for ie in ies]
        ies, ie_max, ie_min = self.normalize(ies, positive=True)

        radia_data = {
            'mid': mids,
            'smiles': smis,
            'HOMO (eV)': homos,
            'LUMO (eV)': lumos,
            'HOMO LUMO Gap (eV)': gaps,
            'IE (eV)': ies,
            'Heterosatom': heteros,
            'Dipole Moment': dipoles,
            'Aromatic': aros,
            'Synthesizability': reset_retros,
            '$\lambda_{h}$ (eV)': hres,
            'LogP': logps,
            'TPSA': tpsas
        }

        border_data = {
            'HOMO (eV)': (homo_min, homo_max),
            'LUMO (eV)': (lumo_min, lumo_max),
            'HOMO LUMO Gap (eV)': (gap_min, gap_max),
            'IE (eV)': (ie_min, ie_max),
            'Heterosatom': (0, 1),
            'Dipole Moment': (dipole_min, dipole_max),
            'Aromatic': (aro_min, aro_max),
            'Synthesizability': (0, 2),
            '$\lambda_{h}$ (eV)': (hre_max, hre_min),
            'LogP': (logp_max, logp_min),
            'TPSA': (tpsa_min, tpsa_max)
        }
        labels = ['HOMO (eV)', 'LUMO (eV)', 'HOMO LUMO Gap (eV)', 'IE (eV)', 'Heterosatom', 'Aromatic',
                  'Synthesizability', '$\lambda_{h}$ (eV)']  # 三个维度
        pd.DataFrame(hist_data).to_csv('bar.csv', index=False)
        self.draw_all(hist_data, radia_data, border_data, labels)

        radia_data['Synthesizability'] = retros
        scores = []
        for i in range(len(radia_data['mid'])):
            score = 0
            for key in labels:
                if key != 'Synthesizability':
                    score += radia_data[key][i]
            scores.append(score)
        radia_data['score'] = scores
        radia_df = pd.DataFrame(radia_data)
        radia_df = radia_df.sort_values(by='score', ascending=False).reset_index()
        radia_df.to_csv('radia_without_synthetic.csv', index=False)

        for i in range(4):
            print('=' * 20)
            print(radia_df['smiles'][i])
            for label in labels:
                print(f"{label}: {round(radia_df[label][i], 2)}")


if __name__ == '__main__':
    conf = ConfigUtils.load_config('../config.json').solar_config
    d = Draw(conf)
    d.draw_homo_lumo_distribute()
    d.draw_her_distribute()
    d.draw_retrosyn_distribute()
    d.draw_radia()
    d.draw_final_score()
