from pyspark.sql import *
import pandas as pd
import ML_scalaAdvanceMethods as sparkML

pd.set_option('display.max_columns', 15)
spark = SparkSession.builder.master("local").appName("my App").getOrCreate()
# sLogger.getLogger("org").setLevel(Level.ERROR)

rawDF2 = spark.read.format("csv"). \
    option("Delimiter", ","). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/Users/keeratjohar2305/Downloads/Dataset/AVtrain_LoanPrediction.csv").\
    na.fill("Female", ["Gender"]).\
    na.fill("Yes", ["Married"]).\
    na.fill("2", ["Dependents"]).\
    na.fill("No", ["Self_Employed"]).\
    na.fill(146.4, ["LoanAmount"]).\
    na.fill(90, ["Loan_Amount_Term"]).\
    na.fill(1, ["Credit_History"])
    # withColumn("ApplicantIncome", F.col("ApplicantIncome").cast(DoubleType)).\
    # withColumn("CoapplicantIncome", F.col("CoapplicantIncome").cast(DoubleType)).\
    # withColumn("LoanAmount", F.col("LoanAmount").cast(DoubleType)).\
    # withColumn("Loan_Amount_Term", F.col("Loan_Amount_Term").cast(DoubleType))

rawTestDF = spark.read.format("csv"). \
    option("Delimiter", ","). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/Users/keeratjohar2305/Downloads/Dataset/AVtest_LoanPrediction2.csv")

print(rawDF2.printSchema())
rawDF2.show(100)
#sparkML.summaryCustomized(rawDF2).show()

# Observe care fully the output of summary (outlet size has null values )
    # sparkML.summaryCustomized(rawDF).show()
    # sparkML.dsShape(rawDF)
DiscreteNumericTypeCols = ['Outlet_Establishment_Year']
DiscreteStringTypeCols = ['Outlet_Establishment_Year']

continuousStringTypeCols = ['Loan_ID']
continuousNumericTypeCols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

NominalStringTypeCatCols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
NominalNumericTypeCatCols =[]

OrdinalStringTypeCatCols = []
OrdinalNumericTypeCatCols = ['Dependents','Loan_Amount_Term']


print("Now lets start some panda stuff and plots {Exploratory Data Analysis}")
df2 = rawDF2.toPandas()
df2.dtypes
print(df2.shape)


# Analysis 1.1:  Uni-variate Analysis for Catagorical Columns
sparkML.makeUnivariateCategoricalplotBARV (NominalStringTypeCatCols + OrdinalNumericTypeCatCols, df2)
exit(1)
sparkML.makeUnivariateCategoricalplotBARH (NominalStringTypeCatCols + OrdinalNumericTypeCatCols, df2)
sparkML.makeUnivariateCategoricaPlotPIE (NominalStringTypeCatCols + OrdinalNumericTypeCatCols, df2)

# Analysis 1.2:  Uni-variate Analysis for Catagorical Columns
sparkML.makeUnivariateCategoricalPlotHISTGRAM(continuousNumericTypeCols, df2)

# Analysis 2.1:  Bi-variate Analysis of categorical columns against Label Column
sparkML.crossTabHistGramPlot(NominalStringTypeCatCols + OrdinalNumericTypeCatCols, 'Loan_Status', df2)
sparkML.crossTabHistGramPlotSplit(NominalStringTypeCatCols + OrdinalNumericTypeCatCols, 'Loan_Status', df2)


# Analysis 2.2:  Bi-variate Analysis of categorical columns against Contineous Column
for contineousColname in continuousNumericTypeCols:
    sparkML.makegGroupedBoxPlotVertical(NominalStringTypeCatCols + OrdinalNumericTypeCatCols, contineousColname, df2)



df2['Loan_Status'].replace({'Y': 1, 'N': 0}, inplace=True)
avg_loanGivenBycreaditHistory = df2.groupby('Credit_History')['Loan_Status'].mean()
NumberOf_loansGivenBycreaditHistory = df2.groupby('Credit_History')['Loan_Status'].count()  # df2['Credit_History'].value_counts()
sparkML.makeHistogramForSeries(NumberOf_loansGivenBycreaditHistory, "Applicants by Credit_History", "Credit_History", "Count of Applicants")
sparkML.makeHistogramForSeries(avg_loanGivenBycreaditHistory, "Probability of getting loan by credit history", "Credit_History", "Probability of getting loan")
exit(1)


#print(df.apply(lambda x: sum(x.isnull()), axis=0))
#train['Date.of.Birth']= pd.to_datetime(FinalDF['Date.of.Birth'])
#train['ltv'] = train['ltv'].astype('int64')

"""
Histogram:
Histograms are one of the most common graphs used to display numeric data. 
Histograms two important things we can learn from a histogram:
    distribution of the data — Whether the data is normally distributed or if it’s skewed (to the left or right)
    To identify outliers — Extremely low or high values that do not fall near any other data points.
"""


# print ("------")
# rawDF.groupBy("Outlet_Size", "Outlet_Location_Type").count().createOrReplaceTempView("T1")
# rawDF.groupBy("Outlet_Size", "Outlet_Type").count().createOrReplaceTempView("T2")
# rawDF.groupBy("Outlet_Location_Type", "Outlet_Type").count().createOrReplaceTempView("T3")
# spark.sql("select * from T1").show()
# spark.sql("select * from T2").show()
# spark.sql("select * from T3").show()
#
# T1 -> mediam (930)
# 				GR (528)->   0
# 				M1 (1860) -> 930
#       Small (1458)
# 			    GR(528)->   528
# 			    M1(1860)->  930
# T3-> High(932) ->
# 				GR (555)-> 0
# 				M1 (932)-> 932
# 				M2 (928)-> 0
# 				M3 (935)-> 0
# 	 Mediam(1863)->
# 	 			GR (555)-> 0
# 	 			M1 (932)->
# 	 			M2 (928)-> 928
# 	 			M3 (935)-> 935
#      null(555) ->
# 	 			GR (555)-> 555****     small/large/mediam
# 	 			M1 (932)-> 0
# 	 			M2 (928)-> 0
# 	 			M3 (935)-> 0
# T2 -> Small(930)
# 				M1(2785)
# 	  Null(1855)
# 	  			M1(2785)*****   small/large/mediam
