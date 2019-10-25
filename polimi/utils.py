

__all__ = ['set_rc_defaults', 'ColorFactory']


def set_rc_defaults():
    import matplotlib.pyplot as plt
    plt.rc('font', family='Arial', size=8)
    plt.rc('lines', linewidth=1, color='k')
    plt.rc('axes', linewidth=1, titlesize='medium', labelsize='medium')
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')
    #plt.rc('figure', dpi=300)


class ColorFactory:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    def __init__(self):
        pass
    def __color__(self, col, str):
        return col + str + self.ENDC
    def red(self, str):
        return self.__color__(self.RED, str)
    def green(self, str):
        return self.__color__(self.GREEN, str)
    def blue(self, str):
        return self.__color__(self.BLUE, str)
    def yellow(self, str):
        return self.__color__(self.YELLOW, str)

