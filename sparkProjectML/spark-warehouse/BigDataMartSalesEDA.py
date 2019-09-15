from pyspark.sql import *
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd
import ML_scalaAdvanceMethods as sparkML

pd.set_option('display.max_columns', 15)
spark = SparkSession.builder.master("local").appName("my App").getOrCreate()
# sLogger.getLogger("org").setLevel(Level.ERROR)

rawDF = spark.read.format("csv"). \
    option("Delimiter", ","). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/Users/keeratjohar2305/Downloads/Dataset/AV_trainBigDataMartSales.txt")

rawTestDF = spark.read.format("csv"). \
    option("Delimiter", ","). \
    option("header", "true"). \
    option("inferSchema", "true"). \
    load("/Users/keeratjohar2305/Downloads/Dataset/AV_testBigDataMartSales.txt")

print(rawDF.printSchema())
df = rawDF.toPandas()

print(df.shape)

column = df['Item_MRP']
plt1 = plt
num_bins = 20
plt1.title("Histogram")
plt1.grid(True)
plt1.hist(column, num_bins,  facecolor='blue', alpha=0.5)
plt1.show()

plt1.title("boxPlot")
plt1.grid(True)
plt1.boxplot(column)
plt1.show()

exit(1)



# Observe care fully the output of summary (outlet size has null values )
    # sparkML.summaryCustomized(rawDF).show()
    # sparkML.dsShape(rawDF)

ColumnsContainsNull = ['Item_Weight', 'Outlet_Size']
rawDF.groupBy('Outlet_Size').count().show()

# lets understand the data to build logic to replace null value of Outlet_Size using logic
rawDF.createOrReplaceTempView("rawDF")
rawDF.crosstab('Outlet_Location_Type', 'Outlet_Size').withColumnRenamed("null", "nullVal").createOrReplaceTempView("T1")
rawDF.crosstab('Outlet_Location_Type', 'Outlet_Type').withColumnRenamed("null", "nullVal").createOrReplaceTempView("T2")
rawDF.crosstab('Outlet_Type', 'Outlet_Size').withColumnRenamed("null", "nullVal").createOrReplaceTempView("T3")
spark.sql("select case when T3.High >  T3.Medium  and T3.High >  T3.Small  then 'High'" +
       " when T3.Medium >  T3.High  and T3.Medium >  T3.Small  then 'High'" +
       " else  'Small' end DrivedOutlet, T1.Outlet_Location_Type_Outlet_Size, T1.nullVal " +
       " from T1, T2, T3  where T3.nullVal=T1.nullVal and  " +
       " T1.Outlet_Location_Type_Outlet_Size=T2.Outlet_Location_Type_Outlet_Type and T1.nullVal>0").\
        createOrReplaceTempView("DrivedOutlet")


# lets fill the null of Outlet_Size using logic
FinalDF = spark.sql("select T1.* , case when T1.Outlet_Size is null then T2.DrivedOutlet " +
          "    else T1.Outlet_Size end drivedOutletSize  from rawDF T1 left outer join DrivedOutlet T2 " +
          " on  T2.Outlet_Location_Type_Outlet_Size = T1.Outlet_Location_Type ").drop("Outlet_Size")

## mean values of Item_Weight, We will use to fill the missing values
outletSizeMean = rawDF.select(F.col("Item_Weight")).agg(F.mean("Item_Weight")).collect()[0].asDict().get('avg(Item_Weight)')
FinalDF.withColumn("Item_Weight", F.when(F.isnull(F.col('Item_Weight')), outletSizeMean).otherwise(F.col('Item_Weight')))

## update the columns,Item_Fat_Content to replace the correct uniform values
FinalDF.withColumn("Item_Fat_Content",
                     F.when(F.col('Item_Fat_Content').isin("LF", "low fat"), "Low Fat").\
                     when(F.col('Item_Fat_Content').isin("reg"), "Regular").\
                     otherwise(F.col('Item_Fat_Content')))


#sparkML.summaryCustomized(FinalDF).show()


print("""
    Lets seperate out columns/Variable type
        #Numerical -> discrete / contineous
        #Categorical-> ordinal, nominal""")



catCols = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Establishment_Year", "Outlet_Size",
           "Outlet_Location_Type", "Outlet_Type"]
DiscreteNumbericTypeCols = ['Outlet_Establishment_Year']
contineusStringTypeCols = ['Item_Identifier']
contineusNumericTypeCols = ['Item_Weight', 'Item_Visibility', 'Item_MRP']

catNominalStringTypeCols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type']
catOrdinalStringTypeCols = ['drivedOutletSize']


print("Now lets start some panda stuff and plots {Exploratory Data Analysis}")

df = FinalDF.toPandas()
print(df.head(10))
# print(df.describe())

#print(df.apply(lambda x: sum(x.isnull()), axis=0))
#train['Date.of.Birth']= pd.to_datetime(FinalDF['Date.of.Birth'])
#train['ltv'] = train['ltv'].astype('int64')



for colname in DiscreteNumbericTypeCols:
    print("start for " + colname)
    sparkML.makeUnivariateCategoricalPlot(colname, df)

for colname in catNominalStringTypeCols:
    print("start for " + colname)
    sparkML.makeUnivariateCategoricalPlot(colname, df)

for colname in catOrdinalStringTypeCols:
    print("start for " + colname)
    sparkML.makeUnivariateCategoricalPlot(colname, df)


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
