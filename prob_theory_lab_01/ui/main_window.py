# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_main_window(object):
    def setupUi(self, main_window):
        main_window.setObjectName("main_window")
        main_window.resize(1076, 659)
        self.centralwidget = QtWidgets.QWidget(main_window)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.num_of_observ_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.num_of_observ_in.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.num_of_observ_in.setFont(font)
        self.num_of_observ_in.setObjectName("num_of_observ_in")
        self.gridLayout_3.addWidget(self.num_of_observ_in, 6, 1, 1, 1)
        self.calc_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.calc_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.calc_btn.setFont(font)
        self.calc_btn.setObjectName("calc_btn")
        self.gridLayout_3.addWidget(self.calc_btn, 7, 1, 1, 1)
        self.q_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.q_in.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.q_in.setFont(font)
        self.q_in.setObjectName("q_in")
        self.gridLayout_3.addWidget(self.q_in, 3, 1, 1, 1)
        self.n_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.n_in.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.n_in.setFont(font)
        self.n_in.setObjectName("n_in")
        self.gridLayout_3.addWidget(self.n_in, 5, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 4, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 3, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 6, 0, 1, 1)
        self.r_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.r_in.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.r_in.setFont(font)
        self.r_in.setObjectName("r_in")
        self.gridLayout_3.addWidget(self.r_in, 4, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 5, 0, 1, 1)
        self.plot_btn = QtWidgets.QPushButton(self.groupBox_2)
        self.plot_btn.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.plot_btn.setFont(font)
        self.plot_btn.setObjectName("plot_btn")
        self.gridLayout_3.addWidget(self.plot_btn, 10, 1, 1, 1)
        self.bin_edges_table = QtWidgets.QTableWidget(self.groupBox_2)
        self.bin_edges_table.setMinimumSize(QtCore.QSize(100, 0))
        self.bin_edges_table.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.bin_edges_table.setFont(font)
        self.bin_edges_table.setObjectName("bin_edges_table")
        self.bin_edges_table.setColumnCount(1)
        self.bin_edges_table.setRowCount(6)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        self.bin_edges_table.setItem(5, 0, item)
        self.gridLayout_3.addWidget(self.bin_edges_table, 9, 1, 1, 1)
        self.m_rows_in = QtWidgets.QLineEdit(self.groupBox_2)
        self.m_rows_in.setMaximumSize(QtCore.QSize(120, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.m_rows_in.setFont(font)
        self.m_rows_in.setObjectName("m_rows_in")
        self.gridLayout_3.addWidget(self.m_rows_in, 8, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 1, 0, 3, 1)
        self.untitled_table = QtWidgets.QTableWidget(self.centralwidget)
        self.untitled_table.setMinimumSize(QtCore.QSize(0, 105))
        self.untitled_table.setMaximumSize(QtCore.QSize(16777215, 105))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.untitled_table.setFont(font)
        self.untitled_table.setObjectName("untitled_table")
        self.untitled_table.setColumnCount(0)
        self.untitled_table.setRowCount(2)
        item = QtWidgets.QTableWidgetItem()
        self.untitled_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.untitled_table.setVerticalHeaderItem(1, item)
        self.gridLayout.addWidget(self.untitled_table, 3, 1, 1, 1)
        self.plot = MplWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.plot.setFont(font)
        self.plot.setObjectName("plot")
        self.gridLayout.addWidget(self.plot, 1, 1, 1, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setMaximumSize(QtCore.QSize(16777215, 100))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox_3.setFlat(False)
        self.groupBox_3.setCheckable(False)
        self.groupBox_3.setObjectName("groupBox_3")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.num_chars_table = QtWidgets.QTableWidget(self.groupBox_3)
        self.num_chars_table.setMinimumSize(QtCore.QSize(690, 56))
        self.num_chars_table.setMaximumSize(QtCore.QSize(685, 56))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.num_chars_table.setFont(font)
        self.num_chars_table.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.num_chars_table.setObjectName("num_chars_table")
        self.num_chars_table.setColumnCount(8)
        self.num_chars_table.setRowCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setItalic(False)
        item.setFont(font)
        self.num_chars_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.num_chars_table.setHorizontalHeaderItem(7, item)
        self.num_chars_table.horizontalHeader().setDefaultSectionSize(85)
        self.horizontalLayout.addWidget(self.num_chars_table)
        self.gridLayout.addWidget(self.groupBox_3, 2, 1, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 110))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 0, 0, 1, 2)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setMaximumSize(QtCore.QSize(187, 9999999))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.frame.setFont(font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.table = QtWidgets.QTableWidget(self.frame)
        self.table.setMinimumSize(QtCore.QSize(160, 0))
        self.table.setMaximumSize(QtCore.QSize(187, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.table.setFont(font)
        self.table.setObjectName("table")
        self.table.setColumnCount(1)
        self.table.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.table.setHorizontalHeaderItem(0, item)
        self.table.horizontalHeader().setDefaultSectionSize(120)
        self.verticalLayout.addWidget(self.table)
        self.label_6 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout.addWidget(self.label_6)
        self.max_sub_th_stat_pdf = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.max_sub_th_stat_pdf.setFont(font)
        self.max_sub_th_stat_pdf.setText("")
        self.max_sub_th_stat_pdf.setAlignment(QtCore.Qt.AlignCenter)
        self.max_sub_th_stat_pdf.setObjectName("max_sub_th_stat_pdf")
        self.verticalLayout.addWidget(self.max_sub_th_stat_pdf)
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.verticalLayout.addWidget(self.label_7)
        self.d_out_lbl = QtWidgets.QLabel(self.frame)
        self.d_out_lbl.setText("")
        self.d_out_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.d_out_lbl.setObjectName("d_out_lbl")
        self.verticalLayout.addWidget(self.d_out_lbl)
        self.check_hypothesis_btn = QtWidgets.QPushButton(self.frame)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.check_hypothesis_btn.setFont(font)
        self.check_hypothesis_btn.setObjectName("check_hypothesis_btn")
        self.verticalLayout.addWidget(self.check_hypothesis_btn)
        self.gridLayout.addWidget(self.frame, 0, 2, 4, 1)
        main_window.setCentralWidget(self.centralwidget)

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslateUi(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "Лабораторная работа по теории вероятностей"))
        self.groupBox_2.setTitle(_translate("main_window", "Параметры задачи"))
        self.num_of_observ_in.setText(_translate("main_window", "100"))
        self.calc_btn.setText(_translate("main_window", "PushButton"))
        self.q_in.setText(_translate("main_window", "1"))
        self.n_in.setText(_translate("main_window", "5"))
        self.label_3.setText(_translate("main_window", "R"))
        self.label_2.setText(_translate("main_window", "Q"))
        self.label_4.setText(_translate("main_window", "n"))
        self.r_in.setText(_translate("main_window", "0.5"))
        self.label_5.setText(_translate("main_window", "N"))
        self.plot_btn.setText(_translate("main_window", "Графики"))
        item = self.bin_edges_table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Интервал"))
        __sortingEnabled = self.bin_edges_table.isSortingEnabled()
        self.bin_edges_table.setSortingEnabled(False)
        item = self.bin_edges_table.item(0, 0)
        item.setText(_translate("main_window", "3"))
        item = self.bin_edges_table.item(1, 0)
        item.setText(_translate("main_window", "4"))
        item = self.bin_edges_table.item(2, 0)
        item.setText(_translate("main_window", "5"))
        item = self.bin_edges_table.item(3, 0)
        item.setText(_translate("main_window", "7"))
        item = self.bin_edges_table.item(4, 0)
        item.setText(_translate("main_window", "9"))
        item = self.bin_edges_table.item(5, 0)
        item.setText(_translate("main_window", "12"))
        self.bin_edges_table.setSortingEnabled(__sortingEnabled)
        self.m_rows_in.setText(_translate("main_window", "6"))
        item = self.untitled_table.verticalHeaderItem(0)
        item.setText(_translate("main_window", "stat_pdf"))
        item = self.untitled_table.verticalHeaderItem(1)
        item.setText(_translate("main_window", "th_pdf"))
        self.groupBox_3.setTitle(_translate("main_window", "Числовые характеристики"))
        item = self.num_chars_table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Eη"))
        item = self.num_chars_table.horizontalHeaderItem(1)
        item.setText(_translate("main_window", "🆇"))
        item = self.num_chars_table.horizontalHeaderItem(2)
        item.setText(_translate("main_window", "|Eη - 🆇|"))
        item = self.num_chars_table.horizontalHeaderItem(3)
        item.setText(_translate("main_window", "Dη"))
        item = self.num_chars_table.horizontalHeaderItem(4)
        item.setText(_translate("main_window", "S^2"))
        item = self.num_chars_table.horizontalHeaderItem(5)
        item.setText(_translate("main_window", "|Dη - S^2|"))
        item = self.num_chars_table.horizontalHeaderItem(6)
        item.setText(_translate("main_window", "🅼e"))
        item = self.num_chars_table.horizontalHeaderItem(7)
        item.setText(_translate("main_window", "🆁"))
        self.groupBox.setTitle(_translate("main_window", "Условие задачи"))
        self.label.setText(_translate("main_window", "<html><head/><body><p>Устройство состоит из <span style=\" font-style:italic;\">N</span> &gt;&gt; 1 дублирующих приборов. Каждый следующий прибор включается после выхода из строя предыдущего. Время безотказной работы каждого прибора - положительная с.в. со средним <span style=\" font-style:italic;\">Q </span>и дисперсией <span style=\" font-style:italic;\">R. </span>Плотность распределения выбрать по аналогии с приведенными на рисунках на с. 15. С.в. <span style=\" font-style:italic;\">η - </span>время безотказной работы всего устройтва.</p></body></html>"))
        item = self.table.horizontalHeaderItem(0)
        item.setText(_translate("main_window", "Время работы"))
        self.label_6.setText(_translate("main_window", "<html><head/><body><p>max|stat_pdf - th_pdf| ↓</p></body></html>"))
        self.label_7.setText(_translate("main_window", "<html><head/><body><p><span style=\" font-size:10pt;\">D ↓</span></p></body></html>"))
        self.check_hypothesis_btn.setText(_translate("main_window", "Гипотеза"))
from mpl_widget import MplWidget
