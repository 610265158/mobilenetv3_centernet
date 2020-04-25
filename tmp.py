#-*-coding:utf-8-*-






def find(x,target):

    if len(x)==1:
        return x
    mid=(len(x)+1)//2


    if x[mid]>target:
        return find(x[0:mid],target)
    elif x[mid]<target:
        return find(x[mid:],target)
    else:
        return [x[mid]]



test1=[1,2,3,4,5,6]

print(test1.index(find(test1,6)[0])+1)

test2=[5,8,9,33,555,33333,2141241]

print(test2.index(find(test2,6)[0])+1)

test3=[8,8,9,33,555,33333,2141241]
print(test3.index(find(test3,2141246)[0])+1)