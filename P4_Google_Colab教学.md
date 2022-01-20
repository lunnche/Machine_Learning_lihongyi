## Google Colab Tutorial  

## Introduction  

Colaboratory,or "Colab" for short,allows you to write and execute Python in your browser,with  
* Zero configuration required  
* Free access to GPUs  
* Easy sharing  


Colab Demo:https://reurl.cc/ra63jE  

in this demo,you will learn the following:  
* Use the GPU resource provided by google  
* Download files using colab  
* Connect google colab with your google drive  


You can type python code in the code block,or use a leading exclamation mark! to change the code block to treating the input as a shell script  

python:  
```python
import numpy
import math
...
```


shell script:  
```
!ls -l
```

Exclamation mark(!) starts a new shell,does the operations,and then kills that shell,while percentage(%) affects the process associated with the notebook,and it is called a magic function.  

Use % instead of ! for cd(change directory) command  

## Changing Runtime  

To utilize the free GPU provided by google,click on "Runtime"（执行阶段）->"Change Runtime Type"（变更执行阶段类型）.There are three options under "Hardware Accelerator"（硬体加速器）,select "GPU".
*Doing this will restart the session,so make sure you change to the desired runtime before executing any code.  

## Executing Code Block  
Click on the play button to execute the code block.This code downloads a file from google drive  

File Structure  

Clicking on the folder icon will give you the visualization of the file structure  
There should be a jpg file,if you do not see it ,click the refresh button  
The file is temporarily stored ,and will be removed once you end your session.You can download the file to your local directory.  

## Mounting Google Drive  
Execute the code block with drive.mout('/content/drive')  

or click on the Google Drive icon,a code block will appear  

Sign in to your google account to get the authorization code.Enter the authorization code in the box below.  

Execute the following three code blocks in order  

This will download the image to your google drive,so you can access it later  
You can create a new code block by clicking on +Code(程式码）on the top   

Move cell up  
Move cell down  
Delete cell  

## Saving Colab  
You can download the ipynb file to your local device(File>Download.ipynb),or save the colab notebook to your google drive (File>Save a copy in Drive).  

## Recovering Files in Google Drive   
Right Click on File>Manage Versions（版本管理）to recover old files that have been accidentally overwriten.  
  

## Useful Linux Commands(in Colab)  

ls:List all files in the current directory  
ls -l:List all files in the current directory with more detail  
pwd: Output the working directory  
mkdir`<dirname>`:Create a directory named  `<dirname>`  
gdown:Download files from google drive  
wget:Download files from the internet  
`python <python_fiel>`:Executes a python file  

到5：45  


