from fastapi import FastAPI

app = FastAPI()

@app.get("/")     # @함수를 불러오는 함수
def read_root():
    return {"message" : "Hello, FastAPI"}    # key-value 방식 dictionary

@app.get("/item/{item_id}")
def read_item(item_id):
    return{"item_id":item_id}

@app.get("/items/")
def read_items(skip=0, limit=10):            # 이 값을 따로 쓰지않으면 default 는 0 과 10이다.
    return {'skip': skip, "limit": limit}     
