def featurize(data):
    new_data = []
    for d in data:
        tokens = d.split(" ")
        label = int(tokens[0])
        feature_vals = []
        for token in tokens:
            subtokens = token.split(":")
            if len(subtokens) == 1:
                continue
            feature_vals.append(int(subtokens[1]))
        new_data.append([feature_vals, label])

    return new_data
