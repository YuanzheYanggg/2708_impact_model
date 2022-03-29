import os
import subprocess

relative_path = "."
TAQ_trade_path = "data/trades"
TAQ_quote_path = "data/quotes"

'''
Since we are given both trade and quote data for universal tickers within time range between 2007-06 to 2007-09
Each day is compressed as a tar.gz file
What we really want to analyze is the binary file included in those compressed files
So we need to extract daily ticker information by unzip the tar.gz file
Doing it in a manual way is time-consuming
So I write up this helper func to help us automate the process
'''


def Unzip_files(path_name):
    print("All the file path under this dir are: ")
    main_dir = os.getcwd()
    dir_list = os.listdir(os.path.join(relative_path,path_name))
    whole_dir = os.path.join(main_dir,path_name)
    print(whole_dir)
    command = "cd " + whole_dir
    print(command)
    subprocess.call(command, shell=True)
    # we need to relocate the destination dir
    os.chdir(whole_dir)
    # use subprocess to call shell command in order to automate the process
    for dir in dir_list:
        if dir[-6:] == "tar.gz":
            print(dir)
            # use shell command tar -zxvf filename.tar.gz to uncompress the file to current dir
            command = "tar -zxvf " + dir
            # call shell command using subprocess module
            subprocess.call(command, shell=True)
    return


if __name__ == "__main__":
    #Unzip_files(TAQ_trade_path)
    Unzip_files(TAQ_quote_path)