from pyspark.sql import SparkSession

# from pyspark import SparkConf, SparkContext
# conf = SparkConf().setMaster("local").setAppName("FriendsByAge")
# sc = SparkContext(conf = conf)


def parseLine(line):
    fields = line.split(",")
    age = int(fields[2])
    numFriends = int(fields[3])
    return (age, numFriends)


def main():
    spark = SparkSession.builder.master("local").appName("FriendsByAge").getOrCreate()
    sc = spark.sparkContext
    lines = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/fakefriends.csv")
    rdd = lines.map(parseLine)
    totalsByAge = rdd.mapValues(lambda x: (x, 1)).reduceByKey(
        lambda x, y: (x[0] + y[0], x[1] + y[1])
    )
    averagesByAge = totalsByAge.mapValues(lambda x: x[0] / x[1])
    results = averagesByAge.collect()
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
