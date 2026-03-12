from string import punctuation

def keyword_search(query :str, database :dict, stopwords :list[str]) -> list[dict]:
    results = []
    if len(database["movies"]) == 0:
        raise ValueError("database argument has no key 'movies'")
    
    transform = str.maketrans("", "", punctuation)
    tquery = str.translate(query.lower(), transform).split(" ")
    
    for t in tquery:
        if t in stopwords or t == "":
            tquery.remove(t)

    for movie in database["movies"]:
        try:
            # assumes item will be a string;
            # I doubt I need to care, but try block just in case

            tmovie = str.translate(movie["title"].lower(), transform).split(" ")
            for t in tmovie:
                if t in stopwords or t == "":
                    tmovie.remove(t)

            for qt in tquery:
                if movie in results:
                    break
                for mt in tmovie:
                    if qt in mt:
                        results.append(movie)
                        break

        except Exception as e:
            # if corrupt entry isn't a string, just keep going
            continue
    return results

