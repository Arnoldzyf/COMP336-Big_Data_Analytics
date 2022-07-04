from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import math
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import datetime
from datetime import datetime, timedelta

dataset_path = 'dataset.txt'

spark = SparkSession.builder.master("local[1]") \
	    .appName("comp529assign2").getOrCreate()
spark.sparkContext.setLogLevel('WARN')

df = spark.read.option("header",True).csv(dataset_path)

# df.printSchema()
# print((df.count(), len(df.columns))) #(2537330, 8)
# df.show(5)

## 1
print('\n******* 1 *******')

'''
## It's strange that the following code cannot give the correct answer
## It seems that some of the Time may become None

## combine Date and Time to form a new column DateTime
temp = df.withColumn('DateTime', concat(col('Date'),lit(' '), col('Time')))

## turn the new column DateTime into timestamp format
temp = temp.withColumn('DateTime', to_timestamp('DateTime', 'yyyy-MM-dd HH:mm:ss'))

## add 8 hours to DateTime
temp = temp.withColumn('DateTime', temp.DateTime + expr('INTERVAL 8 HOURS'))
'''

def ConvertTime(Date, Time):
    ## combine Date and Time to form n DateTime
    DateTime = '{} {}'.format(Date, Time)
    ## turn it into timestamp format
    DateTime = datetime.strptime(DateTime, "%Y-%m-%d %H:%M:%S")
    ## add 8 hours to DateTime
    DateTime = DateTime + timedelta(hours=8)
    return DateTime.strftime('%Y-%m-%d %H:%M:%S')
 
ConvertTimeUDF = udf(ConvertTime)

## turn the new column DateTime into timestamp format
temp = df.withColumn('DateTime', ConvertTimeUDF(col('Date'), col('Time')))

## update Date and Time based on the new DateTime
temp = temp.withColumn("Date", date_format('DateTime', 'yyyy-MM-dd'))
temp = temp.withColumn('Time', date_format('DateTime', 'HH:mm:ss'))

## add 8 hors to Timestamp
temp = temp.withColumn('Timestamp', col('Timestamp')+8/24)

## delete the DateTime Column
df1 = temp.drop('DateTime')
df1.show(5)

## turn into proper datatypes
df1 = df1.withColumn('Latitude', col('Latitude').cast('double'))
df1 = df1.withColumn('Longitude', col('Longitude').cast('double'))
df1 = df1.withColumn('Altitude', col('Altitude').cast('double'))
df1 = df1.withColumn('Date', to_date(col('Date'), 'yyyy-MM-dd'))

## 2
print('\n******* 2 *******')
## group by UserID, count different date
df2 = df1.groupBy("UserID").agg(countDistinct("Date").alias('Date_count'))
## first sort by Date_count descendingly, and then sort by UserID ascendingly
df2 = df2.orderBy(col('Date_count').desc(),col("UserID").asc())
df2.show(5)

## 3
print('\n******* 3 *******')
## group by UserID and Date, and then count the data points
df3 = df1.groupBy("UserID","Date").count()
## filter the ones with at least 100 data points, and then count date for each user
df3 = df3.where(col("count") >= 100).groupBy("UserID").count()
## sort by UserID for easy checking
df3 = df3.orderBy(col("UserID").asc())
# print(df3.count())
df3.show(df3.count()) # 29

## 4
print('\n******* 4 *******')
## find the highest Altitude for each person
df4 = df1.groupBy("UserID").agg(max('Altitude').alias('Altitude'))
## find the corresponding date(s) at which a person reach the Altitude
df4 = df4.join(df1, on=['UserID','Altitude'], how='left')
## choose the smallest date
df4 = df4.groupBy("UserID", 'Altitude').agg(min('Date').alias('Date'))
## sort by Altitude (and UserID)
df4 = df4.orderBy(col('Altitude').desc(),col("UserID").asc())
df4.show(5)

## 5
print('\n******* 5 *******')
## find the max and min Timestamp for each person
df5 =  df1.groupBy("UserID") \
		.agg(max('Timestamp').alias('max_T'), \
		     min('Timestamp').alias('min_T'))
## calculate the Difference and delete un-needed columns		     
df5 = df5.withColumn('Timespan', col('max_T')-col('min_T')).select(col("UserID"), col('Timespan'))
## sort by Difference (and UserID)
df5 = df5.orderBy(col('Timespan').desc(),col("UserID").asc())
df5.show(5)		     

## 6
print('\n******* 6 *******')
## calculate distance between two points
## Reference: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
def distance(lat1,lon1,lat2,lon2):
	# approximate radius of earth in km
	R = 6373.0

	lat1 = math.radians(lat1)
	lon1 = math.radians(lon1)
	lat2 = math.radians(lat2)
	lon2 = math.radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

	d = R * c
	return d

DistanceUDF=udf(distance, DoubleType())

## partition by UserID and Date   
windowSpec = Window.partitionBy("UserID", 'Date').orderBy('Timestamp')
windowSpecAgg  = Window.partitionBy("UserID", 'Date')
## selected needed columns and add row number
df6 = df1.select(col('UserID'), col('Date'), col('Timestamp'), \
		 col('Latitude').alias("lat2"), col('Longitude').alias("lon2"))
df6 = df6.withColumn("row",row_number().over(windowSpec))

## lag the positoin
df6 = df6.withColumn("lat1",lag("lat2",1).over(windowSpec))
df6 = df6.withColumn("lon1",lag("lon2",1).over(windowSpec))
## delete the null values
df6 = df6.dropna()

## calculate distance and sum the daily distance
df6 = df6.withColumn('Distance', DistanceUDF(col('lat1'),col('lon1'),col('lat2'),col('lon2')))
df6 = df6.withColumn('Daily_dist', sum(col('Distance')).over(windowSpecAgg)) \
		.where(col("row")==2).select('UserID','Date','Daily_dist')

## find the longest daily distance for each user
windowSpec = Window.partitionBy("UserID").orderBy(col('Daily_dist').desc(),col("Date").asc())
df6_1 = df6.withColumn("row",row_number().over(windowSpec)).where(col("row")==1).drop('row')
## sort by UserID for easy checking
df6_1 = df6_1.orderBy(col("UserID").asc())
print('For each user the (earliest) day they travelled the most:')
df6_1.show(df6_1.count())

## sum all the daily distance
n= df6.agg({'Daily_dist': 'sum'}).collect()[0][0]
print('\ntotal distance: '+str(n)+' km')
#m=df6.rdd.map(lambda x: (1,x[2])).reduceByKey(lambda x,y: x + y).collect()[0][1]
#print(m)

print('\n******* end *******')










 
                    
