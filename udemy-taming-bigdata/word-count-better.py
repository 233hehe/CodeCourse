import re
from pyspark.sql import SparkSession


def normalizeWords(text):
    return re.compile(r'\W+', re.UNICODE).split(text.lower())


def main():
    spark = SparkSession.builder.master("local[*]").appName("word_count2").getOrCreate()
    sc = spark.sparkContext
    book = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/Book.txt")
    words = book.flatMap(normalizeWords)
    wordcounts = words.countByValue()

    for word, count in wordcounts.items():
        cleanWord = word.encode("ascii", "ignore")
        if cleanWord:
            print(f"{cleanWord.decode()}, {count}")


if __name__ == '__main__':
    main()
