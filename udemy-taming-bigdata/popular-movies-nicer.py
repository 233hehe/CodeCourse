from pyspark.sql import SparkSession

# from pyspark import SparkConf, SparkContext


def loadMovieNames():
    movieNames = {}
    with open("/home/williamzy11/codebase/SparkCourse/ml-100k/u.item") as f:
        for line in f:
            fields = line.split("|")
            movieNames[int(fields[0])] = fields[1]
    return movieNames


def main():
    spark = SparkSession.builder.master("local").appName("movies").getOrCreate()
    sc = spark.sparkContext

    nameDict = sc.broadcast(loadMovieNames())
    lines = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/ml-100k/u.data")
    movies = lines.map(lambda x: (int(x.split()[1]), 1))
    movieCounts = movies.reduceByKey(lambda x, y: x + y)
    flipped = movieCounts.map(lambda x: (x[1], x[0]))
    sortedMovies = flipped.sortByKey()

    sortedMoviesWithNames = sortedMovies.map(
        lambda countMovie: (nameDict.value[countMovie[1]], countMovie[0])
    )

    results = sortedMoviesWithNames.collect()

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
