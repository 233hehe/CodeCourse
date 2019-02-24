import re
from pyspark.sql import SparkSession


def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())


def main():
    spark = SparkSession.builder.master("local[*]").appName("word_count2").getOrCreate()
    sc = spark.sparkContext
    book = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/Book.txt")
    words = book.flatMap(normalizeWords)

    wordCounts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
    wordCountsSorted = wordCounts.map(lambda x: (x[1], x[0])).sortByKey()
    results = wordCountsSorted.collect()

    for result in results:
        count = str(result[0])
        word = result[1].encode('ascii', 'ignore')
        if (word):
            print(word.decode() + ":\t\t" + count)


if __name__ == '__main__':
    main()
