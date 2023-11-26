# Documentation for Running Test Code

If you wish to run the deprecated version of the final code, please follow the instructions below. This version has been deprecated to make it feasible to run on laptops with limited resources.

1. Setup and Downloading Code:

a. Prerequisites:

A Python IDE, preferably VS Code or PyCharm.
A Linux or a Unix-like operating system. If you're using Windows, you can install the Windows Subsystem for Linux (WSL) from here. Mac users don't need to install anything since MacOS is Unix-based.

b. Downloading the Code:

    The necessary files can be downloaded from either this Google Drive Link or the GitHub repository.
    To clone the repository directly into your Python environment, run the following command:


    git clone https://github.com/JV11x/Wound_FL_Diss_Project.git

c. Dependencies:

    The code requires the FLOWER library. Ensure you have at least Python 3.8 (though 3.10 or newer is recommended). Check your Python version with:


python --version

If you need to upgrade, install the required Python version.
Ensure pip is installed. Install the necessary dependencies with:

    pip install -r requirements.txt

2. Unzipping Files:

a. If you don't have the zip library, install it using:

pip install zip

b. Navigate to the Test_Code directory and run the following to unzip the Medetec_foot_ulcer_224.zip file:
    
    unzip Medetec_foot_ulcer_224.zip

c. Change permissions on the unzipped files with:
    
    sudo chmod 755 Medetec_foot_ulcer_224/train 
    sudo chmod 755 Medetec_foot_ulcer_224/test

3. Run Training:

a. If needed, modify the permissions on the run.sh file:
    
    chmod +x run.sh

b. Start the FL training process with:
    
    ./run.sh

After executing, several new files will be generated in the directory, such as log files and .csv files that display training accuracy results. A .pt file, which represents the test data segment, will also be created. These files provide insights into the training process and its accuracy.

4. Run Testing:

To execute the test script, run:

    python test.py
