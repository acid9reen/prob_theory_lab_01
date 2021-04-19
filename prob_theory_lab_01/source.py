import os
import sys
from dataclasses import dataclass, astuple
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, vectorize, float64, int32  # type: ignore
from PyQt5 import QtWidgets

from main_window import Ui_main_window


@vectorize([float64(float64, float64, float64)])
def v_prob_dist(u: float, l: float, h: float) -> float:
    return -(1 / l) * np.log(u) + h


@njit([float64(float64, float64, int32)])
def s(l: float, h: float, n: int) -> float:
    return np.sum(v_prob_dist(np.random.uniform(0, 1, n), l, h))


@njit([float64[:](float64, float64, int32, int32)])
def calc(l: float, h: float, num_of_observ: int, n: int) -> np.ndarray:
    res = np.zeros(num_of_observ, dtype=np.float64)

    for i in range(num_of_observ):
        res[i] = s(l, h, n)

    return np.sort(res)


@dataclass
class NumChars:
    th_mean: float
    mean: Any
    th_var: float
    var: Any
    median: float
    sample_range: float


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__))

        self.ui = Ui_main_window()
        self.ui.setupUi(self)
        self.m_rows = int(self.ui.m_rows_in.text())
        self.sample_data: np.ndarray = np.ndarray([])

        self.ui.calc_btn.clicked.connect(self.calc_btn_on_click)
        self.ui.m_rows_in.editingFinished.connect(self.update_bin_edges_table)
        self.ui.plot_btn.clicked.connect(self.plot_hist)

    def update_bin_edges_table(self) -> None:
        self.m_rows = int(self.ui.m_rows_in.text())

        while self.ui.bin_edges_table.rowCount() > 0:
            self.ui.bin_edges_table.removeRow(0)

        for row_ind in range(0, self.m_rows):
            self.ui.bin_edges_table.insertRow(row_ind)

    def get_bin_edges(self) -> np.ndarray:
        bin_edges = np.zeros(self.m_rows, dtype=np.float64)

        for i in range(self.m_rows):
            bin_edges[i] = float(self.ui.bin_edges_table.item(i, 0).text())

        return bin_edges

    def plot_hist(self) -> None:
        plt.close()
        plt.hist(self.sample_data, self.get_bin_edges())
        plt.show()

    def print_to_table(self, arr: np.ndarray) -> None:
        row = 0
        while self.ui.table.rowCount() > 0:
            self.ui.table.removeRow(0)

        for val in arr:
            self.ui.table.insertRow(row)
            self.ui.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{val:.4f}"))
            row += 1

    def calc_num_chars(self, data_sample: np.ndarray) -> NumChars:
        return NumChars(
            th_mean=self.q * self.n,
            mean=np.mean(data_sample),
            th_var=self.r,
            var=np.var(data_sample),
            median=np.median(data_sample),
            sample_range=data_sample[-1] - data_sample[0],
        )

    def print_to_table_num_chars(self, num_chars: NumChars) -> None:
        self.ui.num_chars_table.setItem(
            0, 0, QtWidgets.QTableWidgetItem(f"{num_chars.th_mean:.4f}")
        )
        self.ui.num_chars_table.setItem(
            0, 1, QtWidgets.QTableWidgetItem(f"{num_chars.mean:.4f}")
        )
        self.ui.num_chars_table.setItem(
            0,
            2,
            QtWidgets.QTableWidgetItem(
                f"{abs(num_chars.th_mean - num_chars.mean):.4f}"
            ),
        )
        self.ui.num_chars_table.setItem(
            0, 3, QtWidgets.QTableWidgetItem(f"{num_chars.th_var:.4f}")
        )
        self.ui.num_chars_table.setItem(
            0, 4, QtWidgets.QTableWidgetItem(f"{num_chars.var:.4f}")
        )
        self.ui.num_chars_table.setItem(
            0,
            5,
            QtWidgets.QTableWidgetItem(f"{abs(num_chars.th_var - num_chars.var):.4f}"),
        )
        self.ui.num_chars_table.setItem(
            0, 6, QtWidgets.QTableWidgetItem(f"{num_chars.median:.4f}")
        )
        self.ui.num_chars_table.setItem(
            0, 7, QtWidgets.QTableWidgetItem(f"{num_chars.sample_range:.4f}")
        )

    def calc_btn_on_click(self) -> None:
        self.q = float(self.ui.q_in.text())
        self.r = float(self.ui.r_in.text())
        self.n = int(self.ui.n_in.text())
        self.num_of_observ = int(self.ui.num_of_observ_in.text())

        self.h = self.q - self.r
        self.l = 1 / self.r

        self.sample_data = calc(self.l, self.h, self.num_of_observ, self.n)
        self.print_to_table(self.sample_data)
        self.print_to_table_num_chars(self.calc_num_chars(self.sample_data))


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
