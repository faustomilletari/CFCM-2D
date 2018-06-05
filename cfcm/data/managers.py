def DataManager(transforms):
    data = {}

    for function in transforms:
        data = function(data)

    return data
