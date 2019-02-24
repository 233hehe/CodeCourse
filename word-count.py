from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.master("local[*]").appName("wordcount").getOrCreate()
    sc = spark.sparkContext
    book = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/Book.txt")
    words = book.flatMap(lambda x: x.split())
    wordcounts = words.countByValue()

    for word, count in wordcounts.items():
        cleanWord = word.encode('ascii', 'ignore')
        if cleanWord:
            print(cleanWord.decode() + " " + str(count))


if __name__ == '__main__':
    main()
