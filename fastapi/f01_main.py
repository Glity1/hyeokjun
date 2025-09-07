from fastapi import FastAPI

app = FastAPI()

@app.get("/")                                      # @ 함수에 통상적으로 많이 붙는데 데코레이터 함수 : 함수가 함수를 받아들인다. /는 경로
def read_root():                                   # read_root를 실행시킬때 get함수를 같이 실행시키겠다.
    return {"message" : "Hello, World"}

@app.get("/hello/")                                      
def read_root():                                   # 저장하고 난뒤에 업데이트 get의 ""안에 내용 주소 추가로 붙여주면 업데이트  
    return {"message" : "Hello, Hello, Hello"}

@app.get("/hi/")                                      
def read_root():                                   
    return {"message" : "Hi, Hi, Hi"}