Lab 2: Loading and preparing ICPSR data
================

Downloading data from ICPSR
---------------------------

-   Please **download** and follow the instructions found on this PDF [link](downloading-ICPSR-data.pdf) **before moving on to the next steps.**
-   To be able to "click" on the links, you'll have to download the file.

Once you've downloaded the data from ICPSR:
-------------------------------------------

-   **unzip** the folder
-   **make note of where the *unzipped* folder is saved**.

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

This is where folder location comes in.

We want the "address" of the unzipped folder called, **ICPSR\_33181**, but specifically the locations of **DS0003** and **DS0004**.

Insert/type/paste the location of the datasets into the code below, **but replace** `~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv` and `~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv` **with the location of the files on your computer**.

``` r
ds03 <- fread(file = "~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv", 
    sep = "\t", header = TRUE)

ds04 <- fread(file = "~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv", 
    sep = "\t", header = TRUE)
```

### 3. Merging the two datasets together:

``` r
merged_data <- inner_join(ds03, ds04, by = "PUBID")
```

### 4. For this class, we will be working observations from Texas

So we need to subset the Texas data.

To do this, we will first create a dummy variable to mark whether the observation was from Texas or not.

``` r
merged_data$texas <- 0  ## <-- here we are adding a dummy variable to our dataset, labeled `texas`.  

merged_data$texas[(merged_data$CCSITE == 1 | merged_data$FWSITE == 1 | merged_data$HOSITE == 
    1)] <- 1  ## we are marking texas observations with a `1`.  

merged_data$texas[(merged_data$SITE.x == "CC" | merged_data$SITE.x == "FW" | 
    merged_data$SITE.x == "HO")] <- 1  ## we are marking texas observations with a `1`.  


table(merged_data$texas)  ## To check if it worked. We see that 5331 observations (or rows) were based in Texas.  
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
    "ZPENG", "NOHSGED", "ZNEVMAR", "Z1PARENT", "TCHCNT", "ZACH05", "ZACH612", 
    "ZACH1318", "ZYNG05", "ZRENTALH", "ZPUBLICH", "ZFT", "ZPT", "HRWGCAT", "MNTHEMP", 
    "ZNOMEMP", "ZGT24MEMP", "STRECIP", "LTRECIP", "YR3KEMP", "YR3EARNAV", "YREMP", 
    "YR1KEMP", "pyrearn", "EMPPQ1", "PEARN1", "STRTDUM", "YRREC", "YRKREC", 
    "GYRWL", "NMOFWELF", "YRRFS", "YRKRFS", "GYRFS", "GEMP1417", "EM4Q1417", 
    "KEMP2T17", "GEMP2T17", "EM4Q2T17", "GEMP14_0", "EMPCNT2T17", "EARN1417", 
    "ER4GE10K", "EARN2T17", "ER14GE40", "VREC1417", "KREC2T17", "WLC1417", "WLC2T17", 
    "VRFS1417", "KRFS2T17", "FSC14T17", "FSC2T17", "TOTSTP", "TOTNSTP", "VSTP1T48", 
    "INCC1417", "INCC2T17")
```

Next, we're going to keep only those listed above:

``` r
data <- texas_data[, names(texas_data) %in% variables]
```
