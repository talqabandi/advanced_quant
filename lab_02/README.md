Lab 2: Loading and preparing ICPSR data
================

Downloading data from ICPSR
---------------------------

-   Please **download** and follow the instructions found on this PDF [link](downloading-ICPSR-data.pdf) **before moving on to the next steps.**
    -   To be able to "click" on the links in the PDF, you'll have to download the file.
-   Save the ICPSR download into the folder you've created for this class.

Once you've downloaded the data from ICPSR:
-------------------------------------------

-   **unzip** the folder and make sure you save it into your `working directory`-- in other words, the folder you've created for this class, it'll make loading and cleaning the data in R so much easier and straightforward.

Opening RStudio and setting your working directory
--------------------------------------------------

The first task when we open RStudio is to set our working directory.

-   As covered in labs, navigate in the `Files` window to your folder for this class (you can find the `Files` window or tab in the bottom right window pane of RStudio).
-   Click on the class folder.
-   Then in `More` (which has a blue cog/wheel to the left of it), click `Set as Working Directory`.
-   Just to reiterate, *do* make sure that the unzipped `ICPSR_33181` folder is located in your working directory.
    -   If it isn't, then on your computer, **outside of RStudio**, manually drag and drop the folder into your working directory (a.k.a. the folder you've created for this class).

Loading the data into R
-----------------------

### 1. Installing and using the required packages

We will be using the `data.table`, `magrittr` and `dplyr` packages.

If you haven't installed these packages before, then please run this code first:

``` r
install.packages("data.table")
install.packages("dplyr")
install.packages("magrittr")
```

Once that is done, run this chunk below:

``` r
library(data.table)
library(dplyr)
library(magrittr)
```

### 2. Loading the data into R

Now let's load the data into R. Once you've set your working directory and confirmed that the `ICPSR_33181` folder is saved in your working directory, all you have to run is this:

``` r
# Copy and paste as is.

ds03 <- fread(file = "ICPSR_33181/DS0003/33181-0003-Data.tsv", sep = "\t", header = TRUE)

ds04 <- fread(file = "ICPSR_33181/DS0004/33181-0004-Data.tsv", sep = "\t", header = TRUE)
```

If you have the `ICPSR_33181` saved elsewhere (if you're still very new to R, I do not recommend this), then:

Insert/type/paste the location of the datasets into the code below, **but replace** `~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv` and `~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv` **with the location of the files on your computer**.

``` r
ds03 <- fread(file = "~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv", 
    sep = "\t", header = TRUE)

ds04 <- fread(file = "~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv", 
    sep = "\t", header = TRUE)
```

### 3. Merging the two datasets together:

Once you have completed step 2, you can just copy, paste, and run all the code in step 3 through step 5 as-is. When you get to step 6, however, please read through the notes and comments, and edit your input accordingly.

``` r
merged_data <- inner_join(ds03, ds04, by = "PUBID")
```

### 4. For this class, we will be working observations from Texas

So we need to subset the Texas data.

To do this, we will first create a dummy variable to mark whether the observation was from Texas or not.

``` r
merged_data$texas <- 0  # here we are adding a dummy variable to our dataset, labeled `texas`.  

merged_data$texas[(merged_data$CCSITE == 1 | merged_data$FWSITE == 1 | merged_data$HOSITE == 
    1)] <- 1  # we are marking texas observations with a `1`.  

merged_data$texas[(merged_data$SITE.x == "CC" | merged_data$SITE.x == "FW" | 
    merged_data$SITE.x == "HO")] <- 1  # we are marking texas observations with a `1`.  


table(merged_data$texas)  # To check if it worked. You should see that 5331 observations (or rows) were based in Texas.  
```

    ## 
    ##     0     1 
    ## 21730  5331

Next, we will create a new dataset including *only* Texas data:

``` r
texas_data <- merged_data[merged_data$texas == 1, ]
```

### 5. Keeping relevant variables

For this class, you will not be using all 500 variables.

We've compiled a list of the relevant ones, which we will use to further reduce the columns (i.e. dimensions) in our dataset.

First, we have to create a list of the variables we want, which we will assign the very fitting label of `variables`:

``` r
variables <- c("PUBID", "CCSITE", "FWSITE", "HOSITE", "racode", "RAYEAR", "ZFEMALE", 
    "ZAGELT25", "ZAGE2534", "ZAGEGT35", "ZHISP", "ZBLACK", "ZWHITE", "ZOTHETH", 
    "NOHSGED", "Z1PARENT", "TCHCNT", "ZACH05", "ZACH612", "ZACH1318", "ZYNG05", 
    "ZRENTALH", "ZPUBLICH", "MNTHEMP", "ZNOMEMP", "ZGT24MEMP", "STRECIP", "LTRECIP", 
    "YR3KEMP", "YR3EARNAV", "YREMP", "pyrearn", "EMPPQ1", "PEARN1", "YRREC", 
    "YRKREC", "GYRWL", "YRRFS", "YRKRFS", "GYRFS", "GEMP1417", "EM4Q1417", "KEMP2T17", 
    "GEMP2T17", "EM4Q2T17", "GEMP14_0", "EMPCNT2T17", "EARN1417", "ER4GE10K", 
    "EARN2T17", "ER14GE40", "VREC1417", "KREC2T17", "WLC1417", "WLC2T17", "VRFS1417", 
    "KRFS2T17", "FSC14T17", "FSC2T17", "TOTSTP", "TOTNSTP", "VSTP1T48", "INCC1417", 
    "INCC2T17", "ZCUREMP")
```

``` r
data <- texas_data[, names(texas_data) %in% variables]
# Creating our new, Texas-only dataset, with just the variables we are
# interested in.
```

### 6. Saving the dataset on your computer as a `.csv` file.

Next, we are going to save this cleaned up dataset on your computer as a `.csv` file. That way for future assignments, you'll have the dataset prepared for analysis right away, without having to do all these steps over again.

-   We will use the function `write.csv` to save the data. But **do make sure that you write `.csv` at the end of the folder name**

``` r
# if you've been working in your working directory, and you would like to
# save the file there, then all you have to run is this:
write.csv(data, file = "NAME_IT_WHATEVER_YOU_WANT.csv")
# only do this if you followed the directions outlined in the section titled
# 'Opening RStudio and setting your working directory' at the very top of
# this page.

# if you want to save the dataset somewhere else, then run this code, but
# change the file location to where you'd like the dataset to go (again, I
# only recommend this if you are proficient in R):
write.csv(data, file = "~/LOCATION/YOU/WOULD/LIKE/TO/SAVE/THE/DATA/NAME_IT_WHATEVER_YOU_WANT.csv")
```

### 7. To load the data for future assignments:

Whenever you want to load the data, all you have to do is:

-   Open up RStudio.
-   Set your working directory again (just to make sure that you're working in the folder you've created for this class)
    -   Steps for this are found in the **Opening RStudio and setting your working directory** section above.
-   Then once that is done, all you have to do is run this code:

``` r
# the input written in quotes should be the same filename that you wrote in
# quotes when you ran the chunk above (the write.csv function)
data <- read.csv(file = "NAME_IT_WHATEVER_YOU_WANT.csv")
```
