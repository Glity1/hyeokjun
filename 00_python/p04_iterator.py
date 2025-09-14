list = [1,2,3]
nums = iter(list)
# print(nums.next())  # 파이썬 2.0 문법, 지금은 안됨
print(next(nums))   # 1
print(next(nums))   # 2, next를 기억하고 있기 때문에 다음 값을 반환함
print(next(nums))   # 3
# print(next(nums))   # error, StopIteration