from pyspark import SparkConf, SparkContext


def countCoOccurences(line):
    elements = line.split()
    return (int(elements[0]), len(elements) - 1)


def parseNames(line):
    fields = line.split('"')
    return (int(fields[0]), fields[1].encode("utf8"))


def main():
    conf = SparkConf().setMaster("local").setAppName("PopularHero")
    sc = SparkContext(conf=conf)
    names = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/Marvel-Names.txt")
    namesRdd = names.map(parseNames)

    lines = sc.textFile("file:///home/williamzy11/codebase/SparkCourse/Marvel-Graph.txt")

    pairings = lines.map(countCoOccurences)
    totalFriendsByCharacter = pairings.reduceByKey(lambda x, y: x + y)
    flipped = totalFriendsByCharacter.map(lambda xy: (xy[1], xy[0]))

    mostPopular = flipped.max()

    mostPopularName = namesRdd.lookup(mostPopular[1])[0]

    print(
        str(mostPopularName)
        + " is the most popular superhero, with "
        + str(mostPopular[0])
        + " co-appearances."
    )


if __name__ == "__main__":
    main()
