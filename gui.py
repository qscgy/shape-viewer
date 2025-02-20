import sys
import yaml
from utils import _set_config_defaults
from viewer import run_viewer
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QWidget,
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QListWidget,
    QGridLayout,
    QMainWindow,
    QDialog,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QFormLayout,
    QPushButton,
)

class KeyLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(KeyLabel, self).__init__(*args, **kwargs)
    
    def set_key(self, key):
        self.key_obj = key

class ConfigWindow(QDialog):
    def __init__(self, cfg_file):
        super().__init__()

        self.setWindowTitle("Plot Config")
        
        main_layout = QVBoxLayout()

        # read config with pyyaml
        with open(cfg_file) as f:
            config = yaml.safe_load(f)
        
        _set_config_defaults(config)
        layout_list = []
        self.form_field_dict = {}       # holds all input fields and their associated types
        for k, v in config.items():
            if isinstance(v, dict):
                sub_layout = QVBoxLayout()
                title_label = KeyLabel(k)
                title_label.set_key(k)
                font = title_label.font()
                font.setPointSize(15)
                title_label.setFont(font)
                sub_layout.addWidget(title_label)
                grid = QFormLayout()
                sub_dict = {}
                for k2, v2 in v.items():
                    # parse each field in the dict to set up an input field
                    if isinstance(v2, bool):
                        field = QCheckBox()
                        field.setChecked(v2)
                        _t = bool   # _t is used to set the type of the user input
                    elif isinstance(v2, list):
                        field = QLineEdit()
                        field.setFixedWidth(500)
                        field.setText(", ".join(v2))
                        _t = list
                    else:
                        field = QLineEdit()
                        field.setFixedWidth(500)
                        field.setText(str(v2))
                        _t = type(v2)
                    grid.addRow(KeyLabel(k2), field)
                    sub_dict[k2] = (field, _t)
                sub_layout.addLayout(grid)
                layout_list.append(sub_layout)
                self.form_field_dict[k] = sub_dict
        
        for lo in layout_list:
            main_layout.addLayout(lo)
        
        button_box = QDialogButtonBox()
        button_box.clear()

        plot_button = QPushButton("Plot")
        button_box.addButton(plot_button, QDialogButtonBox.ButtonRole.AcceptRole)
        plot_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        button_box.addButton(cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        cancel_button.clicked.connect(self.reject)
        main_layout.addWidget(button_box)
        
        self.setLayout(main_layout)
        self.final_config = {}
    
    def accept(self):
        config = {}
        for key in self.form_field_dict:
            value = self.form_field_dict[key]
            if isinstance(value, dict):
                sub_dict = {}
                for k2 in value:
                    field, _t = value[k2]
                    if isinstance(field, QLineEdit):
                        if _t==list:
                            items = field.text().split(',')
                            sub_dict[k2] = [i.strip() for i in items]
                        else:
                            sub_dict[k2] = _t(field.text())
                    elif isinstance(field, QCheckBox):
                        sub_dict[k2] = field.isChecked()
                config[key] = sub_dict
        self.final_config = config
        super().accept()


if __name__=="__main__":
    app = QApplication(sys.argv)
    window = ConfigWindow('shapes/bell.yaml')
    retval = window.exec()
    cfg = None
    if retval:
        cfg = window.final_config
    app.quit()

    if cfg is not None:
        run_viewer(cfg)