from pyspark.sql import *
import pandas as pd
import ML_scalaAdvanceMethods as sparkML
pd.set_option('display.max_columns', 15)
spark = SparkSession.builder.master("local").appName("my App").getOrCreate()
#sLogger.getLogger("org").setLevel(Level.ERROR)

rawDF = spark.read.format("csv").\
        option("Delimiter", ",").\
        option("header", "true").\
        option("inferSchema", "true").\
        load("/Users/keeratjohar2305/Downloads/Dataset/AV_trainBigDataMartSales.txt")

rawTestDF = spark.read.format("csv").\
            option("Delimiter", ",").\
            option("header", "true").\
            option("inferSchema", "true").\
            load("/Users/keeratjohar2305/Downloads/Dataset/AV_testBigDataMartSales.txt")

print(rawDF.printSchema())
sparkML.summaryCustomized(rawDF).show()
sparkML.dsShape(rawDF)

print("Now lets start some panda stuff and plots {Exploratory Data Analysis}")
catCols = ["Item_Fat_Content", "Item_Type", "Outlet_Identifier", "Outlet_Establishment_Year", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]
catNumbericTypeCols = ['Outlet_Establishment_Year', 'Outlet_Size']
catStringTypeCols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type']
contineusNumericTypeCols = ['Item_Weight', 'Item_Visibility', 'Item_MRP']
contineusStringTypeCols = ['Item_Identifier']

df = rawDF.toPandas()
print(df.head(10))
print(df.describe())

print(df.shape)

print("df.dtypes -------------------\n ", df.dtypes)
print("df.T -------------------\n ", df.T)
print("df.sort_index(axis=1, ascending=False) -------------------\n ", df.sort_index(axis=1, ascending=False))
print("df.sort_values(by='B') -------------------\n ", df.sort_values(by='Item_Fat_Content'))
print("""------------------------------------
                                        df.columns
------------------------------------------------------\n """,
      df.columns)


print("""\n ---------------------------- 
                                df[0:3]  slicing the rows
--------------------------------\n """,
      df[0:3])

print("""\n --------------------------------
                     df.loc[0:2, ['Item_Fat_Content', 'Item_Identifier']] 
                     slicing the rows with selected labels
--------------------------------\n """,
      #Selecting on a multi-axis by label:
      df.loc[0:2, ['Item_Fat_Content', 'Item_Identifier']])

print("""\n --------------------------------
                      df.iloc[0:10, 3:5])
                     specificing the labels and rows=positions
--------------------------------\n """,
      df.iloc[0:10, 3:5])

print("""\n --------------------------------
                     df[df['Item_Identifier'].isin(['FDA15', 'DRC01'])]
                     specificing the labels and rows=positions
--------------------------------\n """,
      df[df['Item_Identifier'].isin(['FDA15', 'DRC01'])])

print("""\n --------------------------------
                  df.iloc[4]  // sliced the row only
--------------------------------\n """,
      df.iloc[4:7])

print("""\n --------------------------------
                  df.iloc[4]  // sliced the row only and columns
--------------------------------\n """,
      df.iloc[4:7, 0:3])

print("""\n --------------------------------
                 df[df.Item_Type == 'Household']  // filter /where
--------------------------------\n """,
      df[df.Item_Type == 'Household'])

