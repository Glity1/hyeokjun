import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 사용자
# plt.rcParams['font.family'] = 'AppleGothic' # Mac 사용자
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

# 1. train.csv 파일 로드
try:
    train_df = pd.read_csv('./_data/dacon/new/train.csv')
    print("train.csv 파일이 성공적으로 로드되었습니다.")
    print(f"원본 데이터 크기: {train_df.shape}")


    # 2. 'Canonical_Smiles' 컬럼의 중복 제거 (훈련 코드와 동일하게 처리)
    train_df_deduplicated = train_df.drop_duplicates('Canonical_Smiles').reset_index(drop=True)
    inhibition_data = train_df_deduplicated['Inhibition']
    print(f"중복 제거 후 데이터 크기: {train_df_deduplicated.shape}")

    # 3. Inhibition 값의 기술 통계량 출력
    print("\n--- inhibition 값의 기술 통계량 ---")
    print(inhibition_data.describe())
    
    # 왜도와 첨도 계산
    print(f"\nInhibition 값의 왜도 (Skewness): {inhibition_data.skew():.4f}")
    print(f"Inhibition 값의 첨도 (Kurtosis): {inhibition_data.kurt():.4f}")
    print("-" * 40)

    # 4. 히스토그램 시각화
    plt.figure(figsize=(10, 6)) # 그래프 크기 설정
    sns.histplot(inhibition_data, bins=50, kde=True, color='skyblue', edgecolor='black')
    
    # 그래프 제목 및 축 라벨 설정
    plt.title('Inhibition 값의 분포', fontsize=16)
    plt.xlabel('Inhibition (%)', fontsize=13)
    plt.ylabel('빈도', fontsize=13)
    
    # x축 범위 설정 (0% ~ 100% 명확히 보기 위해)
    plt.xlim(0, 100) 
    
    # 그리드 추가
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 그래프 보여주기
    plt.show()

except FileNotFoundError:
    print("\n오류: 'train.csv' 파일을 찾을 수 없습니다.")
    print("현재 스크립트가 실행되는 디렉토리에 파일이 있는지 확인하거나, 파일 경로를 정확히 지정해주세요.")
except KeyError:
    print("\n오류: 'train.csv' 파일에 'Inhibition' 컬럼이 없습니다.")
    print("컬럼 이름이 정확한지 확인해주세요.")
except Exception as e:
    print(f"\n데이터를 처리하는 중 예기치 않은 오류가 발생했습니다: {e}")