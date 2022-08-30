def lyndon_words(depth, dim):

    """Generate Lyndon words of length `depth` over an `dim`-symbol alphabet

    Args:
        depth: int
        dim: int
    """

    word = [-1]
    while word:
        word[-1] += 1
        yield word
        m = len(word)
        while len(word) < depth:
            word.append(word[-m])
        while word and word[-1] == dim - 1:
            word.pop()


if __name__ == "__main__":

    import signatory

    depth = 2
    dim = 3

    for w in lyndon_words(depth=depth, dim=dim):
        print(w)

    print(signatory.lyndon_brackets(channels=dim, depth=depth))
