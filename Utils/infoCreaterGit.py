import os

class BasicInfoCreater:
    """
    A class to create Git logs and patches for job submissions.
    """

    GITPATCH = 'gitDiff.patch'

    def __init__(self, log_file_name="summary.dat", summary=""):
        self.CMSSWDirPath = os.environ['CMSSW_BASE']
        self.CMSSWRel = self.CMSSWDirPath.split("/")[-1]
        self.logFileName = log_file_name
        self.summary = summary

    def generate_git_log(self):
        git_log = (
            """CMSSW Version used: {}\n
            Current directory path: {}\n
            Summary for current setup: {}\n""".format(self.CMSSWRel, self.CMSSWDirPath, self.summary)
        )

        with open(self.logFileName, "w") as out_script:
            out_script.write(git_log)
            out_script.write("\n\n============\n== Latest commit summary \n\n\n")
            os.system("git log -1 --pretty=tformat:' Commit: %h %n Date: %ad %n Relative time: %ar %n Commit Message: %s' >> {}".format(self.logFileName))
            out_script.write("\n\n============\n\n")
            os.system("git log -1 --format='%H' >> {}".format(self.logFileName))

    def generate_git_patch(self):
        os.system('git diff > {}'.format(self.GITPATCH))

    def generate_git_patch_and_log(self):
        self.generate_git_patch()
        self.generate_git_log()

    def send_git_log_and_patch_to_eos(self, output_folder):
        print("Copying {} to path: {}".format(self.logFileName, output_folder))
        os.system('cp -f {} {}/{}'.format(self.logFileName, output_folder, self.logFileName))
        print("Copying {} to path: {}".format(self.GITPATCH, output_folder))
        os.system('cp -f {} {}/{}'.format(self.GITPATCH, output_folder, self.GITPATCH))
