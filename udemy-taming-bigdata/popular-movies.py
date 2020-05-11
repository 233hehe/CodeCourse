# from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


def main():
    # conf = SparkConf().setMaster("local").setAppName("PopularMovies")
    # sc = SparkContext(conf = conf)
    sc = SparkSession.builder.master("local").appName("popularmovies").getOrCreate().sparkContext

    lines = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/ml-100k/u.data")
    movies = lines.map(lambda x: (int(x.split()[1]), 1))
    movieCounts = movies.reduceByKey(lambda x, y: x + y)

    flipped = movieCounts.map(lambda xy: (xy[1], xy[0]))
    sortedMovies = flipped.sortByKey()

    results = sortedMovies.collect()

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
