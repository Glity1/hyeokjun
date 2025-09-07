import pandas as pd

data = [
    ['삼성', '1000', '2000'],
    ['현대', '1100', '3000'],
    ['엘지', '2000', '500'],
    ['아모레', '3500', '6000'],
    ['네이버', '100', '1500']
]

index = ['031', '059', '033', '045', '023']
columns = ['종목명', '시가', '종가']

df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)
#      종목명    시가    종가
# 031   삼성    1000    2000
# 059   현대    1100    3000
# 033   엘지    2000     500
# 045  아모레   3500    6000
# 023  네이버    100    1500

print("================================================")
print("시가가 1100원 이상인 행을 모두 출력")

result = []

for i in df.values :
    index = i[0]
    row = (i[1:])
    over_1000 = [x for x in row if int(x) > 1100]
    # print(over_1000)
    index_row = [index] + over_1000
    result.append(index_row)
    
for row in result:
    print(row)   

cond1 = df['시가'] >= '1100'
print(cond1)
# 031    False
# 059     True
# 033     True
# 045     True
# 023    False
print(df[cond1])
print(df[df['시가'] >= '1100'])

#      종목명    시가    종가
# 059   현대  1100  3000
# 033   엘지  2000   500
# 045  아모레  3500  6000

print(df.loc[df['시가'] >= '1100'])


# ✅ for문 + if문 방식
# for i, row in df.iterrows():
#     if row['시가'] >= 1100:
#         print(row)
df3 = df[df['시가'] >= '1100']['종가']   # 이렇게 더 많이 쓴다 df[...]가 더 간단하고 자주 쓰이는 표현
# df3 = df.loc[df['시가'] >= '1100']['종가']

print(df3)
