
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
from cuspy import ConfigUtils


class AnalysisProcess:

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------
    # 1. Pearson 相关
    # ------------------------------------------------------
    @staticmethod
    def analyze_pearson(df, analysis_columns):
        """
        计算 Pearson 相关矩阵并绘制热力图
        """
        df = df[analysis_columns]
        corr = df.corr(method='pearson')
        plt.figure(figsize=(1.2 * len(corr), 0.9 * len(corr)), dpi=300)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True, cbar_kws={'shrink': 1.0}, vmin=-0.6, vmax=1)
        plt.tick_params(axis='both', which='both', length=0)
        plt.title("Pearson Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        return corr

    # ------------------------------------------------------
    # 2. Spearman 相关
    # ------------------------------------------------------
    @staticmethod
    def analyze_spearman(df, analysis_columns):
        df = df[analysis_columns]
        corr = df.corr(method='spearman')
        plt.figure(figsize=(1.2 * len(corr), 0.9 * len(corr)), dpi=300)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True, cbar_kws={'shrink': 1.0}, vmin=-0.6, vmax=1)
        plt.tick_params(axis='both', which='both', length=0)
        plt.title("Spearman Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        return corr

    # ------------------------------------------------------
    # 3. Kendall 相关
    # ------------------------------------------------------
    @staticmethod
    def analyze_kendall(df, analysis_columns):
        df = df[analysis_columns]
        corr = df.corr(method='kendall')
        plt.figure(figsize=(1.2 * len(corr), 0.9 * len(corr)), dpi=300)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", square=True, cbar_kws={'shrink': 0.6}, vmin=-0.6, vmax=1)
        plt.tick_params(axis='both', which='both', length=0)
        plt.title("Kendall Tau Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        return corr

    @staticmethod
    def plot_pairwise_scatter_matrix(df, analysis_columns):
        """
        Pairwise scatter matrix:
        下三角：散点 + 回归线
        上三角：显示 Pearson r
        对角：直方图 + KDE
        """
        data = df[analysis_columns]
        # 使用 PairGrid 自定义
        g = sns.PairGrid(data, vars=analysis_columns, diag_sharey=False, corner=False)
        # 下三角：回归散点
        g.map_lower(sns.regplot, scatter_kws={'s': 18, 'alpha': 0.7}, line_kws={'color': 'red'})
        # 对角：直方图 + KDE
        g.map_diag(sns.histplot, kde=True, bins=20, color="#4C72B0")

        # 设置回归散点图的 x 和 y 轴标签字体大小
        for ax in g.axes.flatten():
            if ax is not None:  # 有些可能是空的
                ax.set_xlabel(ax.get_xlabel(), fontsize=20)  # 设置 x 轴标签字体大小
                ax.set_ylabel(ax.get_ylabel(), fontsize=20)  # 设置 y 轴标签字体大小

        # 上三角：相关系数文本
        def corr_func(x, y, **kws):
            r, _ = stats.pearsonr(x, y)
            ax = plt.gca()
            ax.annotate(f"r = {r:.2f}", xy=(0.5, 0.5), xycoords='axes fraction',
                        ha='center', va='center', fontsize=20)
            ax.set_axis_off()

        g.map_upper(corr_func)
        plt.figure(figsize=(10, 10), dpi=300)
        plt.suptitle("Pairwise Scatter Matrix", y=1.02)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def analyze_distance_correlation(df, analysis_columns):
        """
        距离相关（Distance Correlation）:
        度量任意依赖（非线性/非单调）。
        若安装 dcor 则可用；否则使用简单实现（O(n^2)）。
        返回：距离相关矩阵
        """
        data = df[analysis_columns]
        X = data.values
        cols = data.columns
        n = X.shape[1]
        try:
            import dcor
            def dcor_pair(a, b):
                return dcor.distance_correlation(a, b)
        except ImportError:
            # 简单实现
            def _center_dist_mat(v):
                A = np.abs(v[:, None] - v[None, :])
                n_ = A.shape[0]
                J = np.eye(n_) - np.ones((n_, n_)) / n_
                return J @ A @ J

            def dcor_pair(a, b):
                A = _center_dist_mat(a)
                B = _center_dist_mat(b)
                dcov2 = (A * B).mean()
                dvar_x = (A * A).mean()
                dvar_y = (B * B).mean()
                if dvar_x <= 1e-15 or dvar_y <= 1e-15:
                    return 0.0
                return np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = dcor_pair(X[:, i], X[:, j])
                mat[i, j] = mat[j, i] = val
        dcor_df = pd.DataFrame(mat, index=cols, columns=cols)

        plt.figure(figsize=(1.2 * 8, 0.9 * 8), dpi=300)
        sns.heatmap(dcor_df, vmin=-0.6, vmax=1, cmap="vlag", annot=True, fmt=".2f",
                    square=True, cbar_kws={'label': 'Distance Correlation'})
        plt.tick_params(axis='both', which='both', length=0)
        plt.title("Distance Correlation Heatmap")
        plt.tight_layout()
        plt.show()
        return dcor_df

    def process(self):
        data_df = pd.read_csv(self.config.calc_results_fp)
        # self.analyze_pearson(data_df, self.config.calc_columns)
        # self.analyze_spearman(data_df, self.config.calc_columns)
        # self.analyze_kendall(data_df, self.config.calc_columns)
        self.plot_pairwise_scatter_matrix(data_df, self.config.calc_columns)
        # self.analyze_distance_correlation(data_df, self.config.calc_columns)


if __name__ == '__main__':
    conf = ConfigUtils.load_config('../config.json').sam_data_config
    ap = AnalysisProcess(conf)
    ap.process()
