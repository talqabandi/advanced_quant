Lab 4a: Preparing the data
================

ICPSR
-----

-   We will be working on the ICPSR data that we used in the linear regression homework a few weeks ago.
-   There is one major change to this dataset: We will be adding more variables from the folder `DS0001`.
-   But worry not! I put together the code below which should load the data for you as you run it.

1.  -   First and foremost, set your working directory.

2.  -   Once you do that, make sure that the data you downloaded from ICPSR is in this working directory.

3.  -   If it is not in there, then drag and drop the file manually on your computer (not in RStudio).

It is important that the ICPSR data is in your working directory for the code to run smoothly. If it isn't then you'll just have to replace my file paths with those of your own.

Loading the relevant libraries:
-------------------------------

After setting your working directory, we will need to load in the relevant libraries:

``` r
# DO THESE EVERY TIME YOU OPEN UP R.
options(scipen = 999)
library(data.table)
library(magrittr)
library(dplyr)
library(formatR)
# install.packages('caret')
library(caret)
```

If this does not work, then trying installing the packages via `install.packages()` and run the above code again.

Loading the data into R
-----------------------

### 1. Loading the data into R

Next we will be loading the data into R. If you've set your working directory and confirmed that the `ICPSR_33181` folder is actually in your working directory, all you have to run is this:

``` r
# Copy and paste as is.
ds01 <- fread(file = "ICPSR_33181/DS0001/33181-0001-Data.tsv", sep = "\t", header = TRUE)

ds03 <- fread(file = "ICPSR_33181/DS0003/33181-0003-Data.tsv", sep = "\t", header = TRUE)

ds04 <- fread(file = "ICPSR_33181/DS0004/33181-0004-Data.tsv", sep = "\t", header = TRUE)
```

If you have the `ICPSR_33181` saved elsewhere (if you're still very new to R, I do not recommend this), then:

Insert/type/paste the location of the datasets into the code below, **but replace** `~/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv` **with the location of the files on your computer**.

``` r
ds01 <- fread(file = "D:/Tima/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0001/33181-0001-Data.tsv", 
    sep = "\t", header = TRUE)

ds03 <- fread(file = "D:/Tima/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0003/33181-0003-Data.tsv", 
    sep = "\t", header = TRUE)

ds04 <- fread(file = "D:/Tima/Dropbox/NSSR/2019 Spring/Advanced Quant - TA/ICPSR_33181/DS0004/33181-0004-Data.tsv", 
    sep = "\t", header = TRUE)
```

### 2. Merging the two datasets together:

Now that we've loaded the data, we are going to merge the datasets together. This is very similar to what we did in Lab02, but the difference is that we are now also going to be adding `ds01` to our dataset.

Copy/paste the code as is:

``` r
merged_data <- inner_join(ds03, ds04, by = "PUBID")
merged_data <- inner_join(merged_data, ds01, by = "PUBID")
```

### 3. Texas only.

We are going to need to subset the Texas data.

To do this, we will first create a dummy variable to mark whether the observation was from Texas or not.

``` r
merged_data$texas <- 0  # here we are adding a dummy variable to our dataset, labeled `texas`.  

merged_data$texas[(merged_data$CCSITE == 1 | merged_data$FWSITE == 1 | merged_data$HOSITE == 
    1)] <- 1  # we are marking texas observations with a `1`.  

table(merged_data$texas)
```

    ## 
    ##    0    1 
    ## 5112  775

``` r
merged_data$texas[(merged_data$SITE.x == "CC" | merged_data$SITE.x == "FW" | 
    merged_data$SITE.x == "HO")] <- 1  # we are marking texas observations with a `1`.  

table(merged_data$texas)  # To check if it worked. You should see that 775 observations (or rows) were based in Texas.  
```

    ## 
    ##    0    1 
    ## 5112  775

Next, we will create a new dataset including *only* Texas data:

``` r
texas_data <- merged_data[merged_data$texas == 1, ]
```

### 4. Keeping relevant variables

For this class, you will not be using all of the available variables.

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
    "INCC2T17", "ZCUREMP", "AVEMPPT", "AVEDUPT", "AVEMJS", "AVANYPT", "ASFINT", 
    "ALTCGL", "AENSCH1", "ATCONTCT1", "AHLPCHC", "AHLPHSE", "AHLPTRAN", "AHLPBAS", 
    "AHLPWPRB", "AHLPPERS", "AHIGHGRAD", "E1E3", "E1E4A", "E1E10", "ADLTONLY", 
    "ACURMARY", "AKIDS", "AANYCC", "AANYRCC", "AVPROBCC", "E1F1", "AOWNCAR", 
    "ATAKEBUS", "E1G1", "E1G1A", "ARMEDPRV", "E1H2C", "E1H2D", "E1H2F", "E1H4", 
    "AVHHCS", "AVHHSSI", "AVAPSSI", "AVFEDTAX", "ARESERN", "AOTHERN", "ACHOUS", 
    "ACHOUSTYPE", "E1I1", "E1I2", "E1I4", "AGDHLTH", "ABDHLTH", "AFAMHLTH", 
    "ARESHLTH", "ACHDHLTH", "AOTHHLTH")
```

### 5. Creating our full dataset:

Then we create a dataset including only those variables we want. We are going to call this "full\_data". Because in the next few steps, we will be splitting the dataset into a training and testing set.

Run the code below to create the full dataset:

``` r
full_data <- texas_data[, names(texas_data) %in% variables]
# Creating our new, Texas-only dataset, with just the variables we are
# interested in.
```

### 6. Defining our dependent variable.

For this assignment, the dependent variable is the highest quartile of total welfare use over 4 years.

This is based on the `KREC2T17` variable.

We will need to create a dummy variable for highest total welfare use over 4 years.

First, we need to see what the value of KREC2T17 is at the 75th percentile (a.k.a 3rd quartile). We see that it is 17 months. :

``` r
quantile(full_data$KREC2T17, 0.75, na.rm = TRUE)
```

    ## 75% 
    ##  17

We then use that number (17) as a cutoff for our dummy variable.

We are going to call this new dummy variable, `KREC2T17_cat`. If a person has a KREC2T17 value less than 17, they will be assigned a value of 0 in the new KREC2T17\_cat column. If they have a value equal to or greather than 17, they will be assigned a value of 1 for KREC2T17\_cat.

It is best to do all these lines one by one.

``` r
full_data$KREC2T17_cat[full_data$KREC2T17 < 17] <- 0
full_data$KREC2T17_cat[full_data$KREC2T17 >= 17] <- 1
full_data$KREC2T17_cat <- factor(full_data$KREC2T17_cat)
```

### 7. Splitting our full dataset into a train and test set.

Make sure that `library(caret)` worked when you ran it at the very beginning on this script. You will need it in order to run the chunk below.

``` r
set.seed(89879878)
trainIndex <- createDataPartition(full_data$KREC2T17_cat, p = 0.7, list = FALSE, 
    times = 1)
train_data <- full_data[trainIndex, ]  # Creating our training set
test_data <- full_data[-trainIndex, ]  # creating our testing set
```

### 8. Saving the dataset on your computer as a `.csv` file.

Next, we are going to save these datasets on your computer as a `.csv` file. That way, you'll have them prepared for your random forest assignment, without having to do all these steps over again.

-   We will use the function `write.csv` to save the data. But **do make sure that you write `.csv` at the end of the folder name**

-   If you've been working in your working directory, and you would like to save the file there, then all you have to run is this:

``` r
write.csv(full_data, file = "NAME_IT_WHATEVER_YOU_WANT.csv")
write.csv(train_data, file = "NAME_IT_WHATEVER_YOU_WANT.csv")
write.csv(test_data, file = "NAME_IT_WHATEVER_YOU_WANT.csv")
```

-   If you want to save the dataset somewhere else, then run this code, but change the file location to where you'd like the dataset to go (again, I only recommend this if you are proficient in R):

``` r
write.csv(full_data, file = "~/LOCATION/YOU/WOULD/LIKE/TO/SAVE/THE/DATA/NAME_IT_WHATEVER_YOU_WANT.csv")
write.csv(train_data, file = "~/LOCATION/YOU/WOULD/LIKE/TO/SAVE/THE/DATA/NAME_IT_WHATEVER_YOU_WANT.csv")
write.csv(test_data, file = "~/LOCATION/YOU/WOULD/LIKE/TO/SAVE/THE/DATA/NAME_IT_WHATEVER_YOU_WANT.csv")
```

### 7. To load the data for future assignments:

Whenever you want to load the data, all you have to do is:

-   Open up RStudio.
-   Set your working directory again (just to make sure that you're working in the folder you've created for this class)
-   Then once that is done, all you have to do is run this code:

``` r
# the input written in quotes should be the same filename that you wrote in
# quotes when you ran the chunk above (the write.csv function)
train_data <- read.csv(file = "NAME_IT_WHATEVER_YOU_WANT.csv")
test_data <- read.csv(file = "NAME_IT_WHATEVER_YOU_WANT.csv")
```
