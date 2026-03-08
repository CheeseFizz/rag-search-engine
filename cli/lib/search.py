

def keyword_search(query :str, database :dict):
    results = []
    if len(database["movies"]) == 0:
        raise ValueError("database argument has no key 'movies'")
    for movie in database["movies"]:
        if query in movie["title"]:
            results.append(movie)

    return results

