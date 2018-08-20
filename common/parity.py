#-*- using:utf-8 -*-

# a simple example of def in chap-1. 
def parity(x):
    if x%2 == 0:
        print("偶数")
    elif x%2 == 1:
        print("奇数")
    else:
        print("整数でない")
