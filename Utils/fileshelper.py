import os
from datetime import datetime

CURRENT_DATETIME = datetime.now()

class FileHelper:
    """
    A class to assist in creating directories and paths for log and store areas.
    """
    def __init__(self, log_path, store_area, if_eos=False):
        self.if_eos = if_eos
        self.log_path = log_path
        self.store_area = store_area
        self.dir_name = (
            str(CURRENT_DATETIME.year)[-2:]
            + str(format(CURRENT_DATETIME.month, '02d'))
            + str(format(CURRENT_DATETIME.day, '02d'))
            + "_"
            + str(format(CURRENT_DATETIME.hour, '02d'))
            + str(format(CURRENT_DATETIME.minute, '02d'))
            + str(format(CURRENT_DATETIME.second, '02d'))
        )
        self.eos_string = '/eos/uscms'
        print("==> Time stamp: {}".format(self.dir_name))

    def create_log_dir_with_date(self):
        """
        Create a directory to store log files.
        Returns:
            string -- Path of the created directory
        """
        log_dir_name = os.path.join(self.log_path, self.dir_name)
        if not os.path.exists(log_dir_name):
            os.makedirs(log_dir_name)
        print("==> Created directory for log files: {}".format(log_dir_name))
        return log_dir_name

    def create_store_area(self, path):
        """
        Create a directory in the store area.
        Arguments:
            path {string} -- Name of the directory with path
        Returns:
            string -- Path of the created directory in the store area
        """
        # Reference:https://cernbox.docs.cern.ch/desktop/other-access-methods/eos_xrootd/
        os.system('eos root://eosuser.cern.ch mkdir {path}'.format(path = path))
        print("==> Created directory at eos path: {path}".format(path = path))
        # Add a check to see if the directory was created
        print("==> Checking if the directory was created...")
        if self.if_eos:
            if os.path.exists(self.eos_string + path):
                print("==> Directory was created.")
            else:
                print("==> Directory was not created.")
                exit(1)
        return path

    def create_store_dir_with_date(self, *additional_strings):
        """
        Create the store area directory with a date.
        Arguments:
            additional_strings {tuple} -- Additional directory names to append
        Returns:
            string -- Path of the created directory
        """
        path = self.create_store_area(self.store_area)
        for additional_string in additional_strings:
            if additional_string:
                path = self.create_store_area(os.path.join(path, additional_string))
        path = self.create_store_area(os.path.join(path, self.dir_name))
        return path

    def create_dir_with_date(self):
        """
        Create both log and store area directories.
        Returns:
            string -- Path of the created log directory
            string -- Path of the created store area directory
        """
        log_dir_name = os.path.join(self.log_path, self.dir_name)
        store_area_dir_name = os.path.join(self.store_area, self.dir_name)
        os.makedirs(log_dir_name, exist_ok=True)
        self.create_store_area(self.store_area)
        self.create_store_area(store_area_dir_name)
        print("==> Created directory for log files: {log_dir_name}".format(log_dir_name = log_dir_name))
        print("==> Created directory at eos path: {store_area_dir_name}".format(store_area_dir_name = store_area_dir_name))
        return log_dir_name, store_area_dir_name
