class Father:
    def __init__(self, name):   # init이라는 함수는 제일 먼저 실행됨 /// self는 init함수 그 자체 (그냥 써줌)
        self.name = name
        print("Father __init__ 실행됨")
        print(self.name, "아빠")
    babo = 4
aaa = Father("재현")

class Son(Father):  # Son은 Father에게 상속받는다.
    def __init__(self, name):
        print("Son __init__ 시작")
        super().__init__(name)
        print("Son __init__ 끝")
        print(f"바보 + 천채 : {super().babo + self.cheonje}")
        print(f"바보 + 천채 : {self.babo + self.cheonje}")
    cheonje = 5
bbb = Son("흥민")
# result = Father.babo + Son.cheonje  # 실습 : Son에서 babo + cheonje를 출력하기
# print(f"Father의 babo + Son의 cheonje : {result}")

'''
Father __init__ 실행됨
재현 아빠
Son __init__ 시작
Father __init__ 실행됨
흥민 아빠
Son __init__ 끝
바보 + 천채 : 9
바보 + 천채 : 9

'''