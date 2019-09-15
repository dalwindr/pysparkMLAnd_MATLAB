import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoderEstimator, Normalizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def getSparkSessionConnection(app_ame: str):
    spark = SparkSession.builder.master("local").appName(app_ame).getOrCreate()
    spark


def dataFitnessCheck(df: DataFrame):
    print("Checking If null values is there is dataset or not")
    print("		Total Reconds in Dataset is " + df.count() +"   ")
    print("		Total Null reconds in Dataset is " + df.count() - df.na.drop().count() + "  ")


def summaryCustomized(raw_df: DataFrame):
    param_name = "countDistinct"
    mySchemaTemp = list(filter(lambda x: (x[1] != 'timestamp'), raw_df.dtypes))
    mySchema = list(map(lambda z: (z[0]), mySchemaTemp))
    ColumnListWithDistinct_count = [param_name] + mySchema
    WithDistinctCntSummaryDF = raw_df.select([F.countDistinct(c).alias(c) for c in mySchema]).withColumn(
        param_name, F.lit(param_name)).selectExpr(ColumnListWithDistinct_count)

    param_name = "NullValueCount"
    ColumnListWithNullValueCount = [param_name] + mySchema
    ColumnListWithNullValueCountDF = raw_df.select(
        [sum(F.when(isnull(F.col(c)), 1).otherwise(0)).name(c) for c in mySchema]).\
        withColumn(param_name, F.lit(param_name)).selectExpr(ColumnListWithNullValueCount)

    param_name = "variance"
    ColumnListWithVariance = [param_name] + mySchema
    WithVarianceSummaryDF = raw_df.select([F.variance(c).alias(c) for c in mySchema]).\
        withColumn(param_name, F.lit(param_name)).selectExpr(ColumnListWithVariance)

    return raw_df.summary().union(WithDistinctCntSummaryDF).union(WithVarianceSummaryDF).union(ColumnListWithNullValueCountDF)


def dsShape(df: DataFrame):
    columns_cnt = len(df.columns)
    row_cnt = df.count()
    print("Shape(rows = ", row_cnt, " and columns =", columns_cnt, " \n ")

# plt1.subplot(414)
# plt.boxplot(data)
# plt1.show()


def makeUnivariateCategoricaPlotPIE(catColList, dataFrame):

    print(" Lets create pie plot for following columns ")
    print(catColList)

    #sns.countplot(column)
    #sns.boxplot(column)
    plt1 = plt
    j = 420
    i = 1
    plt1.suptitle("Categorical Column analysis with Pie Chart")
    for colname in catColList:
        column = dataFrame[colname]
        data = column.value_counts()
        #N = data.count()
        #x = np.arange(N)
        print("data ", data)
        plt1.subplot(j + i)
        i = i + 1
        #plt1.title("pie Chart")
        #plt1.grid(True)
        plt1.pie(data, labels=data.index, autopct='%1.1f%%')
        plt1.axis('equal')
        plt1.xlabel(colname)
    plt1.show()


def makeContineusNumericVarMultiHISTGRAMpLot(catColList, dataFrame):

    print(" Lets create Vertical HISTOGRAM plot for column " )
    print(catColList)
    #sns.countplot(column)
    #sns.boxplot(column)
    """A histogram is a plot of the frequency distribution of continuous data by splitting it to small equal-sized bins.
        Quantitative data anaylysis (population, occurances)
        Elements are grouped together, so that they are considered as ranges.
        bars can not be reordered
        
        split observation into logical series of intervals called bins. 
        X-axis indicates, independent variables i.e. classes/ categories/ discrete values
        while the y-axis represents dependent variables i.e. occurrences. 
    """
    plt1 = plt
    j = 220
    i = 1
    plt1.suptitle("Univariate Contineous Variable Analyses using HISTOGRAM for following columns ")
    for calname in catColList:
        column = dataFrame[calname]
        num_bins =  50 # column.value_counts().max() / column.count() * 2
        plt1.subplot(i + j)
        i = i + 1
        plt1.hist(column, num_bins,  facecolor='blue', alpha=0.5)
        plt1.title(calname)
        # Use 'density' instead.
    plt1.show()


def makeContineusNumericVarHISTGRAMpLot(catColList, dataFrame):
    print(" Lets create Vertical HISTOGRAM plot for column ")
    print(catColList)
    # sns.countplot(column)
    # sns.boxplot(column)
    """A histogram is a plot of the frequency distribution of numeric array by splitting it to small equal-sized bins.

    """
    plt1 = plt
    j = 220
    i = 1
    plt1.suptitle("Univariate Contineous Variable Analyses using HISTOGRAM for following columns ")
    for calname in catColList:
        column = dataFrame[calname]
        num_bins =  50 #column.value_counts().max() / column.count() * 2
        plt1.subplot(i + j)
        i = i + 1
        plt1.xlabel(calname + ' Value')
        plt1.ylabel('Frequency')
        plt1.hist(column, num_bins, facecolor='blue', alpha=0.5)
        #plt1.title(calname)
        # Use 'density' instead.
    plt1.show()


def makeContineusVarMultiLinePlot(catColList, dataFrame):
    print(" Lets create Vertical Multi HISTOGRAM plot for column ")
    print(catColList)
    # sns.countplot(column)
    # sns.boxplot(column)
    """A histogram is a plot of the frequency distribution of numeric array by splitting it to small equal-sized bins.

    """
    plt1 = plt
    j = 220
    i = 1
    plt1.suptitle("Univariate Contineous Variable Analyses using HISTOGRAM for following columns ")
    for calname in catColList:
        column = dataFrame[calname]
        num_bins = 50  # column.value_counts().max() / column.count() * 2
        plt1.subplot(i + j)
        i = i + 1
        plt1.hist(column, num_bins, facecolor='blue', alpha=0.5)
        plt1.title(calname)
        # Use 'density' instead.
    plt1.show()


def makeUnivariateCategoricalplotBARH(catColList, dataFrame):
    print(" Lets create Horizontal BAR plot for column " )
    print(catColList)
    #sns.countplot(column)
    #sns.boxplot(column)
    plt1 = plt
    plt1.grid(True)
    j = 420
    i = 1
    plt1.suptitle("Univariate Categroical Analysing using Horizontal BAR  for the following Columns")
    for colname in catColList:
        column = dataFrame[colname]
        data = column.value_counts(ascending=True)
        plt1.subplot(i + j)
        i = i + 1
        N = data.count()
        x = np.arange(N)
        print("x ", x)
        print("data ", data)
        colors = np.random.rand(N * 3).reshape(N, -1)
        plt1.barh(x, data, alpha=0.8, color=colors, align='center')#, tick_label=labels)
        plt1.yticks(x, data.index, fontsize=6)
        plt1.title(colname)
        plt1.ylabel("Occurances")
    plt1.show()

def makeUnivariateCategoricalplotBARV(catColList, dataFrame):
    """Bar graph is a pictorial representation of data that uses bars to compare different categories of data.
       Comparison of discrete variables
       Bars do not touch each other, hence there are spaces between bars.
       Elements are taken as individual entities.
       Bars can be reordered
       Each  vertical bar graph represents time series data.
       It contains two axis, where one axis represents the categories and the other axis shows the discrete values of the data
    """
    print(" Lets create Horizontal BAR plot for column " )
    print(catColList)
    #sns.countplot(column)
    #sns.boxplot(column)
    plt1 = plt
    plt1.grid
    j = 420
    i = 1
    plt1.suptitle("Univariate Categroical Analysing using Horizontal BAR  for the following Columns")
    for colname in catColList:
        column = dataFrame[colname]
        data = column.value_counts(ascending=True)
        plt1.subplot(i + j)
        i = i + 1
        N = data.count()
        x = np.arange(N)
        print("x ", x)
        print("data ", data)
        colors = np.random.rand(N * 4).reshape(N, -1)
        plt1.bar(x, data, alpha=1, color=colors)#, tick_label=labels)
        plt1.xticks(x, data.index, fontsize=10, rotation=15)
        plt1.title(colname)
        plt1.ylabel("Occurances")
    plt1.show()

def makeUnivariateContineosPlot(columnName, dataFrame):
    print("making plot for " + columnName)
    column = dataFrame[columnName]

    plt1 = plt
    plt1.grid(True)
    plt1.subplot(122)
    plt1.boxplot(column)
    plt1.title("BoxPlot")

    num_bins = 50
    plt1.grid(True)
    plt1.suptitle(columnName)
    plt1.subplot(121)
    plt1.hist(column, num_bins, facecolor='blue', alpha=0.5)
    plt1.title("Histogram")
    plt1.show()
    #
    # sns.countplot(column)
    # print(sns)
    # sns.boxplot(column)
    # print(sns)

def makegGroupedBoxPlotVertical(ByColumns, rangeColumn, dataFrame):
    # 1. It create box plot for each group of data for the By column
    # 2. for Each group data of group column , it will count number of rangeColumn items based on item value and tell us
    #        which how many items and falling under what range fo values
    #  Box plot is used to identify the outliers
    """ boxplot() function takes
                #  first arguament as the data array to be plotted ,
                #  second argument patch_artist=True , fills the boxplot and
                #  third argument takes the label to be plotted.
    """
    plt1 = plt
    i = 240
    j = 1

    for colname in ByColumns:
        #plt1.subplot(211)
        ax1 = plt1.subplot(i+j)
        i = i + 1
        dataFrame.boxplot(column=rangeColumn, by=colname, patch_artist=True, ax=ax1)
        plt1.xlabel(colname)
        plt1.title("")
    plt1.suptitle("Horzonal Box for Columns = " + rangeColumn)
    plt1.show()

    # column = dataFrame['ApplicantIncome']
    # NotGraduate = dataFrame.groupby(['Education']).get_group('Not Graduate')['LoanAmount']  # .astype(str)
    # Graduate = dataFrame.groupby(['Education']).get_group('Graduate')['LoanAmount']  # .astype(str)
    # print(type(column))
    # print(type(NotGraduate))
    # print(type(Graduate))
    #
    # print("plot")
    # plt1 = plt
    # plt1.grid(True)
    # plt1.subplot(121)
    # plt1.boxplot([NotGraduate, Graduate], patch_artist=True, labels=['NotGraduate', 'Graduate'])

def makegGroupedBoxPlotHortzontal(ByColumns, rangeColumn, dataFrame):
    # 1. It create box plot for each group of data for the By column
    # 2. for Each group data of group column , it will count number of rangeColumn items based on item value and tell us
    #        which how many items and falling under what range fo values
    #  Box plot is used to identify the outliers
    """ boxplot() function takes
                #  first arguament as the data array to be plotted ,
                #  second argument patch_artist=True , fills the boxplot and
                #  third argument takes the label to be plotted.
    """
    plt1 = plt
    i = 240
    j = 1

    for colname in ByColumns:
        #plt1.subplot(211)
        ax1 = plt1.subplot(i+j)
        i = i + 1
        dataFrame.boxplot(column=rangeColumn, by=colname, vert=0, patch_artist=True, ax=ax1)
        plt1.xlabel(colname)
        plt1.title("")
    plt1.suptitle("Horzonal Box for Columns = " + rangeColumn)
    plt1.show()


def makeHistogramForSeries(Series,title,xlabel, ylabel):
    plt1 = plt
    #plt1.figure(figsize=(8, 4))
    plt1.xlabel(xlabel)
    plt1.ylabel(ylabel)
    plt1.title(title)
    Series.plot(kind='bar')
    plt1.show()

def crossTabHistGramPlot(groupColumn, BinaryValColumn, Dataframe):
    plt1 = plt
    i = 240
    j = 1
    plt1.suptitle("Catagorical Cols Analysis with status/label columns " + BinaryValColumn)
    for colname in groupColumn:
        ax1 = plt1.subplot(j+i)
        i = i + 1
        CreditHistory_loanStatus = pd.crosstab(Dataframe[colname], Dataframe[BinaryValColumn])
        CreditHistory_loanStatus.plot(kind='bar', stacked=True, color=['red', 'blue'], grid=False, ax = ax1)
        #plt1.title(colname)
    plt1.show()

def crossTabHistGramPlotSplit(groupColumn, BinaryValColumn, Dataframe):
    plt1 = plt
    i = 240
    j = 1
    plt1.suptitle("Catagorical Cols Analysis with status/label columns " + BinaryValColumn)
    for colname in groupColumn:
        ax1 = plt1.subplot(j+i)
        i = i + 1
        CreditHistory_loanStatus = pd.crosstab(Dataframe[colname], Dataframe[BinaryValColumn])
        CreditHistory_loanStatus.plot(kind='bar', color=['red', 'blue'], grid=False, ax = ax1)
        #plt1.title(colname)
    plt1.show()

# plt.gcf stands for Get Current Figure vs plt.clf CLear Current Figure
# plt.gfa stands for Get Current Axis vs plt.cca CLear Current Axis
# plt.sca Set the current Axes instance to ax.