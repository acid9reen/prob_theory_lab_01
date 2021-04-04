import os
import sys

import numpy as np
from numba import njit, vectorize, float64, int32
from PyQt5 import QtWidgets

from main_window import Ui_main_window


@vectorize([float64(float64, float64, float64)])
def v_prob_dist(u: float, l, h):
    return -(1 / l) * np.log(u) + h


@njit([float64(float64, float64, int32)])
def s(l, h, n):
    return np.sum(v_prob_dist(np.random.uniform(0, 1, n), l, h))


@njit([float64[:](float64, float64, int32, int32)])
def calc(l, h, num_of_observ, n):
    res = np.zeros(num_of_observ, dtype=np.float64)

    for i in range(num_of_observ):
        res[i] = s(l, h, n)

    return np.sort(res)[::-1]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__))

        self.ui = Ui_main_window()
        self.ui.setupUi(self)

        self.ui.calc_btn.clicked.connect(self.calc_btn_on_click)

    def print_to_table(self, arr):
        row = 0
        while self.ui.table.rowCount() > 0:
            self.ui.table.removeRow(0)

        for val in arr:
            self.ui.table.insertRow(row)
            self.ui.table.setItem(row, 0, QtWidgets.QTableWidgetItem(
                f"{val:.4f}"))
            row += 1

    def calc_btn_on_click(self) -> None:
        self.q = float(self.ui.q_in.text())
        self.r = float(self.ui.r_in.text())
        self.n = int(self.ui.n_in.text())
        self.num_of_observ = int(self.ui.num_of_observ_in.text())

        self.h = self.q - self.r
        self.l = 1 / self.r

        res = calc(self.l, self.h, self.num_of_observ, self.n)
        self.print_to_table(res)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
