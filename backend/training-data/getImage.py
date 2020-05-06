import urllib.request

folder = input("Type of Image: ")
imgLinks = open(folder + ".txt", "r")

resumeNumber= 0

counter = 0
for line in imgLinks:
    if counter < resumeNumber:
        counter += 1
        continue
    print("Getting ", folder, " Image #", counter)

    tokens = line.split(".")
    filetype = tokens[len(tokens) - 1].strip("\n")
    try:
        urllib.request.urlretrieve(line, "./"
            + folder + "/" + str(counter) + "." + filetype)
    except:
        print("Something not okay happened lad")
    counter += 1
