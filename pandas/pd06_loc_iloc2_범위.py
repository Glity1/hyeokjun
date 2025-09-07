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

print('========================= 아모레와 네이버의 시가 =========================')
# print(df.iloc[3:][1]) 
print(df.iloc[3:5]['시가']) 
# print(df.iloc[3:5,'시가']) # 에러
print(df.iloc[3:5,1])

# print(df.loc['045':'023'][1]) #에러
# print(df.loc['045':][1]) #에러
print(df.loc['045':]['시가']) # loc에는 index 명만 들어가야함 섞어서 쓸 수 없음.
print(df.loc['045':, '시가']) 

print(df.iloc[3:5].iloc[1]) 
# print(df.iloc[3:5].loc['시가'])  # 에러 쓰기 불편함 컬럼명으로 loc 범위를 잡아줘야함
print(df.loc['045':].iloc[1]) # 에러
# print(df.iloc[3:5,1])