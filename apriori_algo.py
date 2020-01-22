from apyori import apriori

def loaddataset():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
dataset=loaddataset()
result = list(apriori(dataset))
for item in result:
    print(item,end="")