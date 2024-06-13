import urllib.request

import os
if not os.path.isdir("datasets"):
    os.mkdir("datasets")

datasetHtml = urllib.request.urlopen("https://mespadoto.github.io/proj-quant-eval/post/datasets").read()

datasetHtml = str(datasetHtml)

datasetList = datasetHtml.split("</tr>")[1:-1]

for dataset in datasetList:
    header = "<td><a href=\"../../data"
    tail = "\">X.npy</a>"
    qstr = dataset.split("</td>")[3]
    qstr = qstr.replace(header,"https://mespadoto.github.io/proj-quant-eval/data")
    qstr = qstr.replace(tail, "").replace("\\n", "")

    name = qstr.replace("https://mespadoto.github.io/proj-quant-eval/data/", "").replace("/X.npy", "")

    data = urllib.request.urlopen(qstr)

    with open(f'datasets/{name}.npy', 'wb') as fdata:
        for line in data:
            fdata.write(line)