import sys

from PyQt5.QtWidgets import QApplication

from gui import MedicalImageApp



if __name__ == "__main__":


    app = QApplication(sys.argv)


    window = MedicalImageApp()


    window.show()


    sys.exit(
        app.exec_()
    )