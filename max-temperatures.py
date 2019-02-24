# author: yizheng
from pyspark.sql import SparkSession


def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return (stationID, entryType, temperature)


def main():
    spark = SparkSession.builder.master("local").appName("max-temp").getOrCreate()
    sc = spark.sparkContext
    lines = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/1800.csv")
    parsedLines = lines.map(parseLine)
    maxTemps = (parsedLines.
                filter(lambda x: "TMAX" in x[1]).
                map(lambda x: (x[0], x[2])).
                reduceByKey(lambda x, y: max(x,y))
                )
    results = maxTemps.collect()
    for result in results:
        print(result[0] + "\t{:.2f}F".format(result[1]))

if __name__=="__main__":
    main()
