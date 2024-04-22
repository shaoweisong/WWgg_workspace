"""
# How to run:
python3 condor_setup_lxplus.py
"""
import argparse
import os
import sys
import glob
sys.path.append("Utils/.")

from color_style import style

def main(args):

    # Variables from argparse
    submission_name = args.submission_name
    InputFileFromWhereReadDASNames = args.input_FHfile
    InputbkgFile =  ['/eos/user/s/shsong/HiggsDNA/datadriven/DiPhotonJetsBox_2017_cat1_reweighted.parquet','/eos/user/s/shsong/HiggsDNA/datadriven/DiPhotonJetsBox_2017_cat2_reweighted.parquet','/eos/user/s/shsong/HiggsDNA/datadriven/DatadrivenQCD_2017_cat1_reweighted.parquet','/eos/user/s/shsong/HiggsDNA/datadriven/DatadrivenQCD_2017_cat2_reweighted.parquet']
    EOS_Output_path = args.eos_output_path
    condor_file_name = args.condor_file_name
    condor_queue = args.condor_queue
    condor_log_path = args.condor_log_path
    year = args.year
    script_path = os.path.realpath(__file__).split("condor_setup_lxplus.py")[0]
    # Get top-level directory name from PWD
    TOP_LEVEL_DIR_NAME = os.path.basename(os.getcwd())

    if EOS_Output_path == "":
        # Get the username and its initial and set the path as /eos/user/<UserInitials>/<UserName>/flashggws
        username = os.environ['USER']
        user_initials = username[0:1]
        EOS_Output_path = '/eos/user/'+user_initials+'/'+username+'/flashggws/'
    os.system("mkdir "+EOS_Output_path)
    EOS_Output_path += submission_name
    os.system("mkdir "+EOS_Output_path)
    condor_file_name = 'submit_condor_job_'+submission_name

    # Create directories for storing log files and output files at EOS.
    import fileshelper
    dirsToCreate = fileshelper.FileHelper( (condor_log_path + '/condor_logs/'+submission_name).replace("//","/"), EOS_Output_path)
    output_log_path = dirsToCreate.create_log_dir_with_date()
    storeDir = dirsToCreate.create_store_area(EOS_Output_path)
    dirName = dirsToCreate.dir_name
    print("dirName",dirName)

    post_proc_to_run = "post_proc.py"
    command = "python "+post_proc_to_run 


    with open('input_parquet_files/'+InputFileFromWhereReadDASNames) as in_file:
        outjdl_file = open(condor_file_name+".jdl","w")
        outjdl_file.write("+JobFlavour   = \""+condor_queue+"\"\n")
        outjdl_file.write("Executable = "+condor_file_name+".sh\n")
        outjdl_file.write("Universe = vanilla\n")
        outjdl_file.write("Notification = ERROR\n")
        outjdl_file.write("Should_Transfer_Files = YES\n")
        outjdl_file.write("WhenToTransferOutput = ON_EXIT\n")
        outjdl_file.write("x509userproxy = $ENV(X509_USER_PROXY)\n")
        count = 0
        count_jobs = 0
        for lines in in_file:
            print(lines)
            if lines[0] == "#": continue
            count = count +1
            sample_name = (lines.split('/')[-1]).strip()
            print("==> sample_name = ",sample_name)
            Xmass=sample_name.split('M-')[1]
            print("==> Xmass = ",Xmass)
            dir_name = "MX"+Xmass+"_MH125"
            print(lines.strip())
            FHfile_list = glob.glob(lines.strip()+"/*.parquet")
            count_root_files = 0
            FHfile_list.sort(key=lambda x: "merged_nominal.parquet" not in x)
            ########################################
            #
            #      Create output directory
            #
            ########################################
                # output_masspoint_dir = sample_name+os.sep+dirName
            output_path = EOS_Output_path +  os.sep + str(year) + os.sep 
            print("==> dirName = ",dirName)
            print("==> output_path = ",output_path)
            os.system("mkdir "+EOS_Output_path )
            os.system("mkdir "+EOS_Output_path + os.sep + str(year))
            # os.system("mkdir "+EOS_Output_path + os.sep + sample_name+os.sep+dirName)
            # infoLogFiles.send_git_log_and_patch_to_eos(EOS_Output_path + os.sep + sample_name + os.sep + dirName)
            #  print "==> output_path = ",output_path

            ########################################
            #print 'dasgoclient --query="file dataset='+lines.strip()+'"'
            #print "..."
            

            count_root_files = count_root_files+1
            count_jobs = count_jobs+1
            outjdl_file.write("Output = "+"condor_logs"+"/"+submission_name+"/"+dirName+"/"+sample_name+"_$(Process).stdout\n")
            outjdl_file.write("Error  = "+"condor_logs"+"/"+submission_name+"/"+dirName+"/"+sample_name+"_$(Process).err\n")
            outjdl_file.write("Log  = "+"condor_logs"+"/"+submission_name+"/"+dirName+"/"+sample_name+"_$(Process).log\n")
            outjdl_file.write("Arguments = "+str(FHfile_list)+" "+str(InputbkgFile)+" "+script_path+" "+EOS_Output_path+os.sep+str(year)+" \n")
            outjdl_file.write("Queue \n")
            print("Number of files: ",count_root_files)
            print("Number of jobs (till now): ",count_jobs)
        outjdl_file.close();


    outScript = open(condor_file_name+".sh","w");
    outScript.write('#!/bin/bash');
    outScript.write("\n"+'echo "Starting job on " `date`');
    outScript.write("\n"+'echo "Running on: `uname -a`"');
    outScript.write("\n"+'echo "System software: `cat /etc/redhat-release`"');
    outScript.write("\n"+'source /cvmfs/cms.cern.ch/cmsset_default.sh');
    outScript.write("\n"+'source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh');
    outScript.write("\n"+'cp ${3}create_inputfiles_flashggfinalfit_'+submission_name+'.py . ');
    outScript.write("\n"+'mkdir PBDT_HH_FHSL_combine_'+year);
    outScript.write("\n"+'mkdir PBDT_HH_FHSL_combine_'+year+'/flashgginput');
    outScript.write("\n"+'echo "========================================="');
    outScript.write("\n"+'echo "cat reate_inputfiles_flashggfinalfit_'+submission_name+'.py"');
    outScript.write("\n"+'echo "..."');
    outScript.write("\n"+'cat create_inputfiles_flashggfinalfit_'+submission_name+'.py');
    outScript.write("\n"+'echo "..."');
    outScript.write("\n"+'echo "========================================="');
    outScript.write("\n"+'python create_inputfiles_flashggfinalfit_'+submission_name+'.py --inputFHfile ${1} --inputbkgfile ${2}');
    outScript.write("\n"+'echo "====> List root files : " ');
    outScript.write("\n"+'ls ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX*/*.root');
    outScript.write("\n"+'echo "====> copying flashgginput *.root file to stores area..." ');
    outScript.write("\n"+'if ls ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX*/*.root 1> /dev/null 2>&1; then');
    outScript.write("\n"+'    echo "File *.root exists. Copy this directory."');
    outScript.write("\n"+'    echo "cp root to ${4}"');
    outScript.write("\n"+'    cp -r ./PBDT_HH_FHSL_combine_2017/flashgginput/MX*/ ${4}');
    outScript.write("\n"+'fi');
    outScript.write("\n"+'cd ${_CONDOR_SCRATCH_DIR}');
    outScript.write("\n"+'rm create_inputfiles_flashggfinalfit_'+submission_name+'.py');
    outScript.write("\n"+'rm -r PBDT_HH_FHSL_combine_'+year);
    outScript.close();
    os.system("chmod 777 "+condor_file_name+".sh");

    print("\n#===> Set Proxy Using:")
    print("voms-proxy-init --voms cms --valid 168:00")
    print("\n# It is assumed that the proxy is created in file: /tmp/x509up_u138391. Update this in below two lines:")
    print("cp /tmp/x509up_u138391 ~/")
    print("export X509_USER_PROXY=~/x509up_u138391")
    print("\n#Submit jobs:")
    print("condor_submit "+condor_file_name+".jdl")
    #os.system("condor_submit "+condor_file_name+".jdl")

# Below patch is to format the help command as it is
class PreserveWhitespaceFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condor Job Submission", formatter_class=PreserveWhitespaceFormatter)
    parser.add_argument("--submission_name", default="cat12", help="String to be changed by user.")
    # input_FHfile mandatory
    parser.add_argument("--input_FHfile", default='', required=True,  help="Input file from where to read DAS names.")
    parser.add_argument("--eos_output_path", default='/eos/cms/store/group/phys_b2g/shsong/flashggws/', help="Initial path for operations.")
    parser.add_argument("--condor_log_path", default='./', help="Path where condor log should be saved. By default is the current working directory")
    parser.add_argument("--condor_file_name", default='submit_condor_jobs_lnujj_', help="Name for the condor file.")
    parser.add_argument("--condor_queue", default="workday", help="""
                        Condor queue options: (Reference: https://twiki.cern.ch/twiki/bin/view/ABPComputing/LxbatchHTCondor#Queue_Flavours)

                        name            Duration
                        ------------------------
                        espresso            20min
                        microcentury     1h
                        longlunch           2h
                        workday 8h        1nd
                        tomorrow           1d
                        testmatch          3d
                        nextweek           1w
                        """)
    parser.add_argument("--year", default="2017",type=str, help="Year of data taking.")
    parser.add_argument("--transfer_input_files", default=" ", help="Files to be transferred as input.")

    args = parser.parse_args()
    main(args)
