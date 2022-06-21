


def writeToErrorLog(text : str):
    f = open("dataset/error_log.txt", "a")
    f.write(text)
    f.close()