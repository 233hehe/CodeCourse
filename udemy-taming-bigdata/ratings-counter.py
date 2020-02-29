# from pyspark import SparkConf, SparkContext
import collections
from pyspark.sql import SparkSession

# conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
# sc = SparkContext(conf = conf)
spark = SparkSession.builder.master("local").appName("RatingHistogram").getOrCreate()
sc = spark.sparkContext

lines = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/ml-100k/u.data")
ratings = lines.map(lambda x: x.split()[2])
result = ratings.countByValue()

sortedResults = collections.OrderedDict(sorted(result.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
