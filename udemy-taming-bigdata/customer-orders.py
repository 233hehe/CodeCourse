# author: yizheng
from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    sc = spark.sparkContext
    orderrdd = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/customer-orders.csv")
    cust_order = (
        orderrdd.map(lambda line: line.split(","))
        .map(lambda lt: (lt[0], float(lt[2])))
        .reduceByKey(lambda x, y: x + y)
        .sortBy(lambda tup: tup[1], False)
    )
    result = cust_order.collect()
    [print(i) for i in result]


if __name__ == "__main__":
    main()
