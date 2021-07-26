import sys
from dataclasses import dataclass
from typing import Any

from scipy import stats
import numpy as np
from numba import njit, vectorize, float64, int32  # type: ignore
from PyQt5 import QtWidgets

from ui.main_window import Ui_main_window
from ui.hypothesis_dialogue import Ui_Dialog


@vectorize([float64(float64, float64, float64)])
def v_prob_dist(u: float, l: float, h: float) -> float:
    return -(1 / l) * np.log(u) + h


@njit([float64(float64, float64, int32)])
def s(l: float, h: float, n: int) -> float:
    return float(np.sum(v_prob_dist(np.random.uniform(0, 1, n), l, h)))


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


class Dialogue(QtWidgets.QDialog):
    def __init__(
        self, sample_data: np.ndarray, params: dict, initial_conditions: dict
    ) -> None:
        super(Dialogue, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        self.alpha = float(self.ui.alpha_in.text())
        self.num_of_points_of_interval = int(
            self.ui.num_of_points_of_interval_in.text()
        )
        self.sample_data = sample_data
        self.params = params
        self.initial_conditions = initial_conditions
        self.intervals: np.ndarray
        self.q_is: np.ndarray

        self.ui.check_hypothesis_btn.clicked.connect(self.check_hypothesis)
        self.ui.num_of_points_of_interval_in.editingFinished.connect(
            self.fill_intervals
        )

    def calculate_f_0(self, r_0: float, k: int):
        return 1 - stats.chi2.cdf(r_0, k)

    def check_hypothesis(self):
        self.alpha = float(self.ui.alpha_in.text())
        self.intervals = self.get_intervals()
        self.calculate_q_i_and_fill_table()

        r_0 = self.calculate_r_0()
        print(r_0)
        f_r_0 = self.calculate_f_0(r_0, len(self.intervals) - 2)

        self.ui.f_r_0_lbl.setText(f"{f_r_0:.4f}")

        if f_r_0 >= self.alpha:
            self.ui.hipothesis_verdict_lbl.setText("Принята ✔")
        else:
            self.ui.hipothesis_verdict_lbl.setText("Отвергнута ❌")

    def calculate_r_0(self) -> float:
        n_is, __ = np.histogram(self.sample_data, self.intervals)
        n = self.initial_conditions["num_of_observ"]

        r_0 = 0
        for n_i, q_i in zip(n_is, self.q_is):
            r_0 += ((n_i - n * q_i) * (n_i - n * q_i)) / (n * q_i)

        return r_0

    def calculate_q_i_and_fill_table(self) -> None:
        self.q_is = np.zeros(len(self.intervals) - 1)

        while self.ui.q_out_table.columnCount() > 0:
            self.ui.q_out_table.removeColumn(0)

        for i in range(1, len(self.intervals)):
            q_i = stats.gamma.cdf(self.intervals[i], **self.params) - stats.gamma.cdf(
                self.intervals[i - 1], **self.params
            )

            self.q_is[i - 1] = q_i

            self.ui.q_out_table.insertColumn(i - 1)
            self.ui.q_out_table.setItem(
                0, i - 1, QtWidgets.QTableWidgetItem(f"{q_i:.4f}")
            )

    def fill_intervals(self) -> None:
        self.num_of_points_of_interval = int(
            self.ui.num_of_points_of_interval_in.text()
        )

        first = self.sample_data[0]
        last = self.sample_data[-1]
        step = (last - first) / (self.num_of_points_of_interval - 1)

        while self.ui.intervals_table.rowCount() > 0:
            self.ui.intervals_table.removeRow(0)

        elem = first
        for row_ind in range(0, self.num_of_points_of_interval):
            self.ui.intervals_table.insertRow(row_ind)
            self.ui.intervals_table.setItem(
                row_ind, 0, QtWidgets.QTableWidgetItem(f"{elem:.2f}")
            )
            elem += step

        self.intervals = self.get_intervals()

    def get_intervals(self) -> np.ndarray:
        intervals = np.zeros(self.num_of_points_of_interval, dtype=np.float64)

        for i in range(self.num_of_points_of_interval):
            intervals[i] = float(self.ui.intervals_table.item(i, 0).text())

        return intervals


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        self.ui = Ui_main_window()
        self.ui.setupUi(self)
        self.m_rows = int(self.ui.m_rows_in.text())
        self.sample_data: np.ndarray = np.ndarray([])
        self.q = float(self.ui.q_in.text())
        self.r = float(self.ui.r_in.text())
        self.n = int(self.ui.n_in.text())
        self.num_of_observ = int(self.ui.num_of_observ_in.text())

        self.h = self.q - np.sqrt(self.r)
        self.l = 1 / np.sqrt(self.r)

        self.params: dict
        self.initial_conditions = {
            "n": self.n,
            "num_of_observ": self.num_of_observ,
            "l": self.l,
            "h": self.h,
        }

        self.ui.calc_btn.clicked.connect(self.calc_btn_on_click)
        self.ui.m_rows_in.editingFinished.connect(self.update_bin_edges_table)
        self.ui.plot_btn.clicked.connect(self.plot_hist)
        self.ui.check_hypothesis_btn.clicked.connect(self.check_hypothesis)

    def check_hypothesis(self):
        dlg = Dialogue(self.sample_data, self.params, self.initial_conditions)
        dlg.exec()

    def update_bin_edges_table(self) -> None:
        self.m_rows = int(self.ui.m_rows_in.text())

        first = self.sample_data[0]
        last = self.sample_data[-1]
        step = (last - first) / (self.m_rows - 1)

        while self.ui.bin_edges_table.rowCount() > 0:
            self.ui.bin_edges_table.removeRow(0)

        elem = first
        for row_ind in range(0, self.m_rows):
            self.ui.bin_edges_table.insertRow(row_ind)
            self.ui.bin_edges_table.setItem(
                row_ind, 0, QtWidgets.QTableWidgetItem(f"{elem:.2f}")
            )
            elem += step

    def get_bin_edges(self) -> np.ndarray:
        bin_edges = np.zeros(self.m_rows, dtype=np.float64)

        for i in range(self.m_rows):
            bin_edges[i] = float(self.ui.bin_edges_table.item(i, 0).text())

        return bin_edges

    def plot_hist(self) -> None:
        self.ui.plot.canvas.axes[1].clear()
        bin_edges = self.get_bin_edges()

        self.ui.plot.canvas.axes[1].hist(self.sample_data, bin_edges, density=True)

        self.ui.plot.canvas.axes[0].clear()
        self.ui.plot.canvas.axes[0].hist(
            self.sample_data,
            500,
            density=True,
            histtype="step",
            cumulative=True,
            label="Empirical",
        )

        params = {
            "a": self.n,
            "loc": self.h * self.n,
            "scale": 1 / self.l,
        }

        x = np.linspace(
            stats.gamma.ppf(0.01, **params), stats.gamma.ppf(0.99, **params), 100
        )
        self.ui.plot.canvas.axes[0].plot(x, stats.gamma.cdf(x, **params))
        self.ui.plot.canvas.draw()

        self.params = params

        self.fill_untitled_table(bin_edges, params)

        d = 0
        for i in range(len(self.sample_data)):
            d_curr = max(
                (i + 1) / self.num_of_observ - stats.gamma.cdf(self.sample_data[i], **params),
                stats.gamma.cdf(self.sample_data[i], **params) - i / self.num_of_observ)

            if d_curr > d:
                d = d_curr

        self.ui.d_out_lbl.setText(f"{d:.4f}")

    def fill_untitled_table(self, bin_edges: np.ndarray, params: dict) -> None:
        while self.ui.untitled_table.columnCount() > 0:
            self.ui.untitled_table.removeColumn(0)


        hist, bins = np.histogram(self.sample_data, bins=bin_edges)
        hist = hist / self.num_of_observ

        max_sub = 0
        for index, val in enumerate(hist):
            self.ui.untitled_table.insertColumn(index)
            stat_pdf = val / (bins[index + 1] - bins[index])
            self.ui.untitled_table.setItem(
                0, index, QtWidgets.QTableWidgetItem(f"{stat_pdf:.4f}")
            )

            x = bins[index] + (bins[index + 1] - bins[index]) / 2
            th_pdf = stats.gamma.pdf(x, **params)
            self.ui.untitled_table.setItem(
                1, index, QtWidgets.QTableWidgetItem(f"{th_pdf:.4f}")
            )

            sub = abs(stat_pdf - th_pdf)

            if sub > max_sub:
                max_sub = sub

            self.ui.max_sub_th_stat_pdf.setText(f"{max_sub:.4f}")

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
            th_var=self.r * self.r * self.n,
            var=np.var(data_sample),
            median=float(np.median(data_sample)),
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

        self.initial_conditions = {
            "n": self.n,
            "num_of_observ": self.num_of_observ,
            "l": self.l,
            "h": self.h,
        }

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
