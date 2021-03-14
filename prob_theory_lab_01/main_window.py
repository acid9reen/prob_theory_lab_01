import os
import sys

import numpy as np
from PyQt5 import QtWidgets, uic


class Main_window(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(Main_window, self).__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__))

        uic.loadUi(script_dir + os.path.sep + "main_window.ui", self)
        self.calc_btn.clicked.connect(self.calc_btn_on_click)
    
    def print_to_table(self, arr):
        row = 0
        while (self.table.rowCount() > 0):
            self.table.removeRow(0)
        
        for val in arr:
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(
                        f"{val:.4f}"))
            row += 1

    def calc_btn_on_click(self) -> None:
        self.q = float(self.q_in.text())
        self.r = float(self.r_in.text())
        self.n = int(self.n_in.text())
        self.num_of_observ = int(self.num_of_observ_in.text())

        self.h = self.q - self.r
        self.l = 1 / self.r

        def prob_dist(u: float):
            return -(1 / self.l) * np.log(u) + self.h if u >= self.h else 0 # ??
        
        v_prob_dist = np.vectorize(prob_dist)
        res = np.array([])

        for __ in range(self.num_of_observ):
            res = np.append(res, np.sum(v_prob_dist(np.random.uniform(0, 1, self.n))))

        self.print_to_table(np.sort(res)[::-1])


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = Main_window()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
