# MLPipeline

<h2>MLFlow pipeline for D602 assessment</h2>

This is a set of files that together form the template for the pipeline task of the D602 performance assessment.  

Author:  Eric Lagally, eric.lagally@wgu.edu

<h3>Introduction</h3>
The pipeline that students create will consist of three stages:  import data, clean data, and train and test regression model.  These three stages are called from an MLproject file.  The MLproject file allows the caller to specify that only one stage be run (for testing purposes) or to call the entire pipeline.  If the entire pipeline is called, the "main" script (either main.py or main.R) is run, which calls each stage sequentially and handles the parameters appropriately.  If individual stages are tested, the individual scripts are run.

<h3>The files</h3>

<h4>Python version</h4>
Each stage has two files associated with it in this repository, although students will need to create only one, the Python script (.py) for each stage.  I included the Jupyter notebooks here in case they are helpful, and in fact students may create Jupyter notebooks themselves and then export to .py files.

<b>


- import_data_dev.py
- clean_data_dev.py
- poly_regressor_dev.py
</b>
The main <b>MLproject</b> file is called using an mlflow run command at the command line:

`mlflow run -P data="WA_ONTIME.csv" -P num_alphas=30 .`

This command will run all three stages together by running <b>main.py</b> with two parameters, which are passed to the first and third stages, respectively.  

The MLproject file establishes a conda environment in which the experiments ("candidates" in GitLab terminology) are run.  This environment is specified in the <b>pipeline_env.yaml</b> file.

<h4>R version</h4>
Each stage has two files associated with it in this repository, although students will need to create only one, the R script (.R) for each stage.  I included the R Markdown documents here in case they are helpful, and in fact students may create R Markdown documents themselves and then export to .R files. This can be achieved by entering the command:

`knitr::purl(filename)` 

within the RStudio IDE at the prompt.  The `filename` in this command is the name of the .Rmd file that the student wishes to export to a .R file.  Note that the correct working directory must first be entered by using the:

`setwd(dir)`

command within the RStudio environment at the command prompt, where `dir` is the directory that the R MArkdown document is located. <b>Also important to note</b> is that RStudio handles datetimes returned from a function differently from Rscript (used to run.R files at the command prompt).  There are alternating lines of code in the poly_regressor.Rmd code template that allow students to comment out the RStudio version of the datetime handling and uncomment the Rscript version.  There is a comment to this effect in the code template. 

<b>


- import_data_dev.R
- clean_data_dev.R
- poly_regressor_dev.R
</b>

The main <b>MLproject</b> file (which students need to create - this code is not student-facing) is called using an mlflow run command at the command line:

`mlflow run -P data="WA_ONTIME.csv" -P lambdas=30 .`

This command will run all three stages together by running <b>main.R</b> with two parameters, which are passed to the first and third stages, respectively.  

The MLproject file establishes a conda environment in which the experiments ("candidates" in GitLab terminology) are run.  This environment is specified in the <b>pipeline_env_r.yaml</b> file.



The final file included is a data set downloaded from the Bureau of Transportation Statistics website: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr, used for testing purposes:  <b>WA_ONTIME.csv</b>.
