import random

MAX_ZERO = 284315
MAX_ONES = 492

def generateRandomTestSample():
    while True:

        sentinel = False

        zero_list = [random.randint(0,MAX_ZERO-1) for _ in range(50)]
        ones_list = [random.randint(0,MAX_ONES-1) for _ in range(50)]

        copy_zero = zero_list[:]
        copy_ones = ones_list[:]
        
        for _ in range(len(zero_list)):
            check = copy_zero.pop()
            if check in copy_zero or check == 0:
                sentinel = True
            
        if sentinel:
            continue

        for _ in range(len(ones_list)):
            check = copy_ones.pop()
            if check in copy_ones or check == 0:
                sentinel = True
        
        if sentinel:
            continue
        
        break

    return (zero_list, ones_list)