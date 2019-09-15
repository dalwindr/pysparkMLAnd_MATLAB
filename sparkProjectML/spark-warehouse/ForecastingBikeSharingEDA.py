from pyspark.sql import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ML_scalaAdvanceMethods as sparkML

pd.set_option('display.max_columns', 15)
pd.set_option('expand_frame_repr', False)
from datetime import datetime


spark = SparkSession.builder.master("local").appName("my App").getOrCreate()

rawDF3 = spark.read.format("csv").\
                    option("header", "true").\
                    option("inferSchema", "true").\
                    option("Delimiter", ",").\
                    load("/Users/keeratjohar2305/Downloads/Dataset/KAGGLE_TrainBikeSharingDemand.csv")
print(rawDF3.printSchema)
sparkML.dsShape(rawDF3)
#rawDF3.show(5)
#sparkML.summaryCustomized(rawDF3).show()
df = rawDF3.toPandas()

# Renaming some columns so they make more sense
df = df.rename(columns = {'count':'total_count'}) # since count is also a Dataframe func.

# Convert count to float
df.total_count = df.total_count.astype(np.float)

# Extract month, year and hour information from the timestamp
df['hour'] = df['datetime'].astype(np.str).map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).hour)
df['month'] = df['datetime'].astype(np.str).map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).month)
df['year'] = df['datetime'].astype(np.str).map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).year)
df['date'] = df['datetime'].astype(np.str).map(lambda x: (datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).day)
# Get labels for bins based on the lower value of each bin


def get_bins(dat, num_bins):
    bins = np.zeros(num_bins)
    for ct in range(num_bins):
        bins[ct] = np.around(np.min(dat)+ct*((np.max(dat) - np.min(dat))/num_bins),decimals = 1)
    return bins


# Convert day information to weekday (Monday (0) to Sunday (6))
df['weekday'] = df['datetime'].astype(np.str).map(lambda x:(datetime.strptime(x, "%Y-%m-%d %H:%M:%S")).weekday())
df['temp'] = pd.cut(df.temp, bins = 20,labels = get_bins(df.temp,20))
df['atemp'] = pd.cut(df.atemp, bins = 20,labels = get_bins(df.atemp,20))
df['humidity'] = pd.cut(df.humidity, bins = 20,labels = get_bins(df.humidity,20))
df['windspeed'] = pd.cut(df.windspeed, bins = 20,labels = get_bins(df.windspeed,20))

df['temp'] = df.temp.astype(np.float)
df['atemp'] = df.atemp.astype(np.float)
df['humidity'] = df.humidity.astype(np.float)
df['windspeed'] = df.windspeed.astype(np.float)

print(df.head(10))

DiscreteNumericTypeCols = []
DiscreteStringTypeCols = []

continuousStringTypeCols = []
continuousNumericTypeCols = ['casual', 'registered', 'total_count']

NominalStringTypeCatCols = []
NominalNumericTypeCatCols =['hour', 'month', 'year', 'date']

OrdinalStringTypeCatCols = []
OrdinalNumericTypeCatCols = ['temp', 'atemp', 'humidity', 'windspeed','season', 'holiday', 'workingday','weather']

# Analysis 1.2:  Uni-variate Analysis for Catagorical Columns
sparkML.makeContineusNumericVarHISTGRAMpLot(continuousNumericTypeCols, df)
exit(1)

print("lets start analysis")
# # Analysis 2.2:  Bi-variate Analysis of categorical columns against Contineous Column
for contineousColname in continuousNumericTypeCols:
     sparkML.makegGroupedBoxPlotVertical(OrdinalNumericTypeCatCols, contineousColname, df)
     sparkML.makegGroupedBoxPlotVertical(NominalStringTypeCatCols , contineousColname, df)
exit(1)



# Analysis 1.1:  Uni-variate Analysis for Catagorical Columns
sparkML.makeUnivariateCategoricalplotBARV (NominalStringTypeCatCols + NominalNumericTypeCatCols , df)
sparkML.makeUnivariateCategoricalplotBARH (NominalStringTypeCatCols + NominalNumericTypeCatCols , df)
sparkML.makeUnivariateCategoricaPlotPIE (NominalStringTypeCatCols + NominalNumericTypeCatCols , df)

print("lets start analysis")
# Analysis 1.1:  Uni-variate Analysis for Catagorical Columns
sparkML.makeUnivariateCategoricalplotBARV (OrdinalNumericTypeCatCols, df)
sparkML.makeUnivariateCategoricalplotBARH (OrdinalNumericTypeCatCols, df)
sparkML.makeUnivariateCategoricaPlotPIE (OrdinalNumericTypeCatCols, df)

#
# # Analysis 2.1:  Bi-variate Analysis of categorical columns against Binary Label Column
# sparkML.crossTabHistGramPlot(NominalStringTypeCatCols + OrdinalNumericTypeCatCols, 'Loan_Status', df)
# sparkML.crossTabHistGramPlotSplit(NominalStringTypeCatCols + OrdinalNumericTypeCatCols, 'Loan_Status', df)
#
#


#import sys
#print(df.to_csv(sys.stdout))


def prepare_plot_area(ax):
    # Remove plot frame lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # X and y ticks on bottom and left
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# Defining a color pattern based
colrcode = [(31, 119, 180), (255, 127, 14), \
            (44, 160, 44), (214, 39, 40), \
            (148, 103, 189), (140, 86, 75), \
            (227, 119, 194), (127, 127, 127), \
            (188, 189, 34), (23, 190, 207)]

for i in range(len(colrcode)):
    r, g, b = colrcode[i]
    colrcode[i] = (r / 255., g / 255., b / 255.)

# How does the casual and registered count change over the course of a day?
wd_tot = df.groupby(['workingday', 'hour'])['total_count'].mean()
wd_casual = df.groupby(['workingday', 'hour'])['casual'].mean()
wd_reg = df.groupby(['workingday', 'hour'])['registered'].mean()

wd_tot_std = df.groupby(['workingday', 'hour'])['total_count'].std()
wd_casual_std = df.groupby(['workingday', 'hour'])['casual'].std()
wd_reg_std = df.groupby(['workingday', 'hour'])['registered'].std()

fig, axes = plt.subplots(figsize=(12, 8), nrows=1, ncols=2)
plt.sca(axes[0])
plt.fill_between(list(range(24)), wd_tot[1] - wd_tot_std[1], wd_tot[1] + wd_tot_std[1], color=colrcode[0], alpha=0.3)
wd_tot[1].plot(kind='line', label='Working day:tot', color=colrcode[0])

plt.fill_between(list(range(24)), wd_casual[1] - wd_casual_std[1], wd_casual[1] + wd_casual_std[1], color=colrcode[1], alpha=0.3)
wd_casual[1].plot(kind='line', label='Working day:cas', color=colrcode[1])

plt.fill_between(list(range(24)), wd_reg[1] - wd_reg_std[1], wd_reg[1] + wd_reg_std[1], color=colrcode[2], alpha=0.3)
wd_reg[1].plot(kind='line', label='Working day:reg', color=colrcode[2])

plt.legend(loc='upper left', fontsize=15)
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Rental count')
prepare_plot_area(plt.gca())


plt.sca(axes[1])
plt.fill_between(list(range(24)), wd_tot[0] - wd_tot_std[0], wd_tot[0] + wd_tot_std[0], color=colrcode[0], alpha=0.3)
wd_tot[0].plot(kind='line', label='Working day:tot', color=colrcode[0])

plt.fill_between(list(range(24)), wd_casual[1] - wd_casual_std[0], wd_casual[0] + wd_casual_std[0], color=colrcode[1],
                 alpha=0.3)
wd_casual[0].plot(kind='line', label='Working day:cas', color=colrcode[1])

plt.fill_between(list(range(24)), wd_reg[0] - wd_reg_std[0], wd_reg[0] + wd_reg_std[0], color=colrcode[2], alpha=0.3)
wd_reg[0].plot(kind='line', label='Working day:reg', color=colrcode[2])

plt.legend(loc='upper left', fontsize=15)
plt.xlabel('Hour', fontsize=15)
plt.ylabel('Rental count')
prepare_plot_area(plt.gca())
plt.show()