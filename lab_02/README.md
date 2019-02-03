Loading and preparing ICPSR data
================================

Downloading data from ICPSR
---------------------------

-   Please **download** and follow the instructions found on this PDF
    [link](downloading-ICPSR-data.pdf) before moving on to the next
    steps.
-   To be able to "click" on the links, you'll have to download the
    file.

Once you've downloaded the data from ICPSR:
-------------------------------------------

-   **unzip** the folder  
-   **make note of where the *unzipped* folder is saved**.

Loading the data into R
-----------------------

### 1. Installing and using the required packages

We will be using the `data.table`, `magrittr` and `dplyr` packages.

If you haven't installed these packages before, then please run this
code first:

    install.packages("data.table")
    install.packages("dplyr")
    install.packages("magrittr")

Once that is done, run this chunk below:

    library(data.table)
    library(dplyr)
    library(magrittr)

### 2. Loading the data into R

This is where folder location comes in.

We want the "address" of the unzipped folder called, **ICPSR\_33181**,
but specifically the locations of **DS0003** and **DS0004**.

Insert/type/paste the location of the datasets into the code below,
**but replace**
`~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv`
and
`~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv`
**with the location of the files on your computer**.

    ds03 <- fread(file = "~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv", 
        sep = "\t", header = TRUE)

    ds04 <- fread(file = "~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv", 
        sep = "\t", header = TRUE)

### 3. Merging the two datasets together:

    merged_data <- inner_join(ds03, ds04, by = "PUBID")

### 4. For this class, we will be working observations from Texas

So we need to subset the Texas data.

To do this, we will first create a dummy variable to mark whether the
observation was from Texas or not.

    merged_data$texas <- 0  ## <-- here we are adding a dummy variable to our dataset, labeled `texas`.  

    merged_data$texas[(merged_data$CCSITE == 1 | merged_data$FWSITE == 1 | merged_data$HOSITE == 
        1)] <- 1  ## we are marking texas observations with a `1`.  

    table(merged_data$texas)  ## To check if it worked. We see that 5331 observations (or rows) were based in Texas.  

    ## 
    ##     0     1 
    ## 21730  5331

Next, we will create a new dataset including *only* Texas data:

    texas_data <- merged_data[merged_data$texas == 1, ]

### 5. Keeping relevant variables

For this class, you will not be using all 500 variables.

We've compiled a list of the relevant ones, which we will use to further
reduce the columns (i.e. dimensions) in our dataset.

First, we have to create a list of the variables we want, which we will
assign the very fitting label of `variables`:

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

Next, we're going to keep only those listed above:

    data <- texas_data[, names(texas_data) %in% variables]

    summary(data)

    ##     PUBID              GEMP1417         EM4Q1417         KEMP2T17     
    ##  Length:5331        Min.   :  0.00   Min.   :0.0000   Min.   : 0.000  
    ##  Class :character   1st Qu.:  0.00   1st Qu.:0.0000   1st Qu.: 2.000  
    ##  Mode  :character   Median : 50.00   Median :0.0000   Median : 7.000  
    ##                     Mean   : 47.15   Mean   :0.3138   Mean   : 7.477  
    ##                     3rd Qu.:100.00   3rd Qu.:1.0000   3rd Qu.:13.000  
    ##                     Max.   :100.00   Max.   :1.0000   Max.   :16.000  
    ##                                                                       
    ##     GEMP2T17         EM4Q2T17         GEMP14_0        EMPCNT2T17    
    ##  Min.   :  0.00   Min.   :0.0000   Min.   :0.0000   Min.   : 0.000  
    ##  1st Qu.: 12.50   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.: 1.000  
    ##  Median : 43.75   Median :1.0000   Median :0.0000   Median : 3.000  
    ##  Mean   : 46.73   Mean   :0.5779   Mean   :0.1439   Mean   : 3.784  
    ##  3rd Qu.: 81.25   3rd Qu.:1.0000   3rd Qu.:0.0000   3rd Qu.: 5.000  
    ##  Max.   :100.00   Max.   :1.0000   Max.   :1.0000   Max.   :31.000  
    ##                                                                     
    ##     EARN1417        ER4GE10K         EARN2T17         ER14GE40     
    ##  Min.   :    0   Min.   :0.0000   Min.   :     0   Min.   :0.0000  
    ##  1st Qu.:    0   1st Qu.:0.0000   1st Qu.:  1400   1st Qu.:0.0000  
    ##  Median : 1500   Median :0.0000   Median :  9900   Median :0.0000  
    ##  Mean   : 5709   Mean   :0.2255   Mean   : 19436   Mean   :0.1664  
    ##  3rd Qu.: 8850   3rd Qu.:0.0000   3rd Qu.: 28350   3rd Qu.:0.0000  
    ##  Max.   :41900   Max.   :1.0000   Max.   :145800   Max.   :1.0000  
    ##                                                                    
    ##     VREC1417         KREC2T17        WLC1417        WLC2T17     
    ##  Min.   :0.0000   Min.   : 0.00   Min.   :   0   Min.   :    0  
    ##  1st Qu.:0.0000   1st Qu.: 4.00   1st Qu.:   0   1st Qu.:  700  
    ##  Median :0.0000   Median :10.00   Median :   0   Median : 1800  
    ##  Mean   :0.2457   Mean   :13.54   Mean   : 273   Mean   : 2457  
    ##  3rd Qu.:0.0000   3rd Qu.:21.00   3rd Qu.:   0   3rd Qu.: 3700  
    ##  Max.   :1.0000   Max.   :48.00   Max.   :2700   Max.   :12900  
    ##                                                                 
    ##     VRFS1417         KRFS2T17        FSC14T17       FSC2T17     
    ##  Min.   :0.0000   Min.   : 0.00   Min.   :   0   Min.   :    0  
    ##  1st Qu.:1.0000   1st Qu.:20.00   1st Qu.: 100   1st Qu.: 4800  
    ##  Median :1.0000   Median :38.00   Median :2700   Median :10300  
    ##  Mean   :0.7524   Mean   :32.21   Mean   :2777   Mean   :10804  
    ##  3rd Qu.:1.0000   3rd Qu.:47.00   3rd Qu.:4600   3rd Qu.:15700  
    ##  Max.   :1.0000   Max.   :48.00   Max.   :9300   Max.   :34700  
    ##                                                                 
    ##      TOTSTP          TOTNSTP           VSTP1T48         INCC1417    
    ##  Min.   :   0.0   Min.   : 0.0000   Min.   :0.0000   Min.   :    0  
    ##  1st Qu.:   0.0   1st Qu.: 0.0000   1st Qu.:0.0000   1st Qu.: 3200  
    ##  Median :   0.0   Median : 0.0000   Median :0.0000   Median : 6700  
    ##  Mean   : 188.4   Mean   : 0.9013   Mean   :0.1231   Mean   : 8759  
    ##  3rd Qu.:   0.0   3rd Qu.: 0.0000   3rd Qu.:0.0000   3rd Qu.:12300  
    ##  Max.   :2800.0   Max.   :14.0000   Max.   :1.0000   Max.   :46000  
    ##                                                                     
    ##     INCC2T17          CCSITE          FWSITE           HOSITE      
    ##  Min.   :     0   Min.   :0.000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.: 16600   1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:0.0000  
    ##  Median : 28100   Median :0.000   Median :0.0000   Median :0.0000  
    ##  Mean   : 32697   Mean   :0.324   Mean   :0.2949   Mean   :0.3812  
    ##  3rd Qu.: 43000   3rd Qu.:1.000   3rd Qu.:1.0000   3rd Qu.:1.0000  
    ##  Max.   :171500   Max.   :1.000   Max.   :1.0000   Max.   :1.0000  
    ##                                                                    
    ##     racode              RAYEAR        ZFEMALE     ZAGELT25     
    ##  Length:5331        Min.   :2001   Min.   :1   Min.   :0.0000  
    ##  Class :character   1st Qu.:2001   1st Qu.:1   1st Qu.:0.0000  
    ##  Mode  :character   Median :2001   Median :1   Median :0.0000  
    ##                     Mean   :2001   Mean   :1   Mean   :0.4091  
    ##                     3rd Qu.:2002   3rd Qu.:1   3rd Qu.:1.0000  
    ##                     Max.   :2002   Max.   :1   Max.   :1.0000  
    ##                                                                
    ##     ZAGE2534         ZAGEGT35          ZHISP            ZBLACK      
    ##  Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.0000  
    ##  Median :0.0000   Median :0.0000   Median :0.0000   Median :0.0000  
    ##  Mean   :0.3808   Mean   :0.2101   Mean   :0.3812   Mean   :0.4658  
    ##  3rd Qu.:1.0000   3rd Qu.:0.0000   3rd Qu.:1.0000   3rd Qu.:1.0000  
    ##  Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
    ##                                                                     
    ##      ZWHITE          ZOTHETH      ZPENG         NOHSGED      
    ##  Min.   :0.0000   Min.   :0   Min.   : NA    Min.   :0.0000  
    ##  1st Qu.:0.0000   1st Qu.:0   1st Qu.: NA    1st Qu.:0.0000  
    ##  Median :0.0000   Median :0   Median : NA    Median :1.0000  
    ##  Mean   :0.1531   Mean   :0   Mean   :NaN    Mean   :0.5091  
    ##  3rd Qu.:0.0000   3rd Qu.:0   3rd Qu.: NA    3rd Qu.:1.0000  
    ##  Max.   :1.0000   Max.   :0   Max.   : NA    Max.   :1.0000  
    ##                               NA's   :5331                   
    ##     ZNEVMAR        Z1PARENT     TCHCNT          ZACH05      
    ##  Min.   : NA    Min.   :1   Min.   :1.000   Min.   :0.0000  
    ##  1st Qu.: NA    1st Qu.:1   1st Qu.:1.000   1st Qu.:0.0000  
    ##  Median : NA    Median :1   Median :2.000   Median :1.0000  
    ##  Mean   :NaN    Mean   :1   Mean   :1.869   Mean   :0.7153  
    ##  3rd Qu.: NA    3rd Qu.:1   3rd Qu.:3.000   3rd Qu.:1.0000  
    ##  Max.   : NA    Max.   :1   Max.   :3.000   Max.   :1.0000  
    ##  NA's   :5331                                               
    ##     ZACH612          ZACH1318         ZYNG05          ZRENTALH     
    ##  Min.   :0.0000   Min.   :0.000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:0.0000   1st Qu.:0.000   1st Qu.:0.0000   1st Qu.:0.0000  
    ##  Median :0.0000   Median :0.000   Median :1.0000   Median :0.0000  
    ##  Mean   :0.4412   Mean   :0.194   Mean   :0.7153   Mean   :0.2322  
    ##  3rd Qu.:1.0000   3rd Qu.:0.000   3rd Qu.:1.0000   3rd Qu.:0.0000  
    ##  Max.   :1.0000   Max.   :1.000   Max.   :1.0000   Max.   :1.0000  
    ##                                                                    
    ##     ZPUBLICH           ZFT            ZPT          HRWGCAT    
    ##  Min.   :0.0000   Min.   : NA    Min.   : NA    Min.   : NA   
    ##  1st Qu.:0.0000   1st Qu.: NA    1st Qu.: NA    1st Qu.: NA   
    ##  Median :0.0000   Median : NA    Median : NA    Median : NA   
    ##  Mean   :0.2084   Mean   :NaN    Mean   :NaN    Mean   :NaN   
    ##  3rd Qu.:0.0000   3rd Qu.: NA    3rd Qu.: NA    3rd Qu.: NA   
    ##  Max.   :1.0000   Max.   : NA    Max.   : NA    Max.   : NA   
    ##                   NA's   :5331   NA's   :5331   NA's   :5331  
    ##     MNTHEMP        ZNOMEMP         ZGT24MEMP         STRECIP      
    ##  Min.   :1.00   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000  
    ##  1st Qu.:2.00   1st Qu.:0.0000   1st Qu.:0.0000   1st Qu.:0.0000  
    ##  Median :3.00   Median :0.0000   Median :0.0000   Median :0.0000  
    ##  Mean   :3.23   Mean   :0.1405   Mean   :0.2542   Mean   :0.4474  
    ##  3rd Qu.:5.00   3rd Qu.:0.0000   3rd Qu.:1.0000   3rd Qu.:1.0000  
    ##  Max.   :5.00   Max.   :1.0000   Max.   :1.0000   Max.   :1.0000  
    ##                                                                   
    ##     LTRECIP         YR3KEMP        YR3EARNAV         YREMP       
    ##  Min.   :0.000   Min.   : 0.00   Min.   :    0   Min.   :0.0000  
    ##  1st Qu.:0.000   1st Qu.: 2.00   1st Qu.:  400   1st Qu.:0.0000  
    ##  Median :0.000   Median : 6.00   Median : 2300   Median :1.0000  
    ##  Mean   :0.176   Mean   : 6.03   Mean   : 4244   Mean   :0.7132  
    ##  3rd Qu.:0.000   3rd Qu.:10.00   3rd Qu.: 6200   3rd Qu.:1.0000  
    ##  Max.   :1.000   Max.   :12.00   Max.   :25800   Max.   :1.0000  
    ##                                                                  
    ##     YR1KEMP        pyrearn          EMPPQ1           PEARN1    
    ##  Min.   : NA    Min.   :    0   Min.   :0.0000   Min.   :   0  
    ##  1st Qu.: NA    1st Qu.:    0   1st Qu.:0.0000   1st Qu.:   0  
    ##  Median : NA    Median : 1600   Median :0.0000   Median :   0  
    ##  Mean   :NaN    Mean   : 4155   Mean   :0.4641   Mean   : 895  
    ##  3rd Qu.: NA    3rd Qu.: 6300   3rd Qu.:1.0000   3rd Qu.:1300  
    ##  Max.   : NA    Max.   :26700   Max.   :1.0000   Max.   :7400  
    ##  NA's   :5331                                                  
    ##     STRTDUM         YRREC            YRKREC           GYRWL       
    ##  Min.   : NA    Min.   :0.0000   Min.   : 0.000   Min.   :  0.00  
    ##  1st Qu.: NA    1st Qu.:0.0000   1st Qu.: 0.000   1st Qu.:  0.00  
    ##  Median : NA    Median :0.0000   Median : 0.000   Median :  0.00  
    ##  Mean   :NaN    Mean   :0.3437   Mean   : 2.696   Mean   : 60.42  
    ##  3rd Qu.: NA    3rd Qu.:1.0000   3rd Qu.: 5.000   3rd Qu.:200.00  
    ##  Max.   : NA    Max.   :1.0000   Max.   :12.000   Max.   :200.00  
    ##  NA's   :5331                                                     
    ##     NMOFWELF        YRRFS            YRKRFS           GYRFS      
    ##  Min.   : NA    Min.   :0.0000   Min.   : 0.000   Min.   :  0.0  
    ##  1st Qu.: NA    1st Qu.:0.0000   1st Qu.: 0.000   1st Qu.:  0.0  
    ##  Median : NA    Median :1.0000   Median : 6.000   Median :200.0  
    ##  Mean   :NaN    Mean   :0.7211   Mean   : 5.866   Mean   :190.4  
    ##  3rd Qu.: NA    3rd Qu.:1.0000   3rd Qu.:11.000   3rd Qu.:300.0  
    ##  Max.   : NA    Max.   :1.0000   Max.   :12.000   Max.   :500.0  
    ##  NA's   :5331
