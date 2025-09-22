
from cuspy import ConfigUtils
from gaussian_calc import GaussianCalc


class DFTProcess:

    def __init__(self, config):
        self.gaussian_calc = GaussianCalc(config)

    def process(self):
        self.gaussian_calc.prepare_init_files()
        self.gaussian_calc.prepare_neural_to_cation_init_file()
        self.gaussian_calc.prepare_cation_init_file()
        self.gaussian_calc.extract_hole_re()
        self.gaussian_calc.prepare_cation_to_neural_init_file()
        self.gaussian_calc.prepare_init_files()
        self.gaussian_calc.prepare_init_tzvp_files()
        self.gaussian_calc.extract_dipole_moment()


if __name__ == '__main__':
    conf = ConfigUtils.load_config('../config.json').solar_config
    dft_process = DFTProcess(conf)
    dft_process.process()
