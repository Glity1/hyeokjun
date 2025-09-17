#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import os
import datetime
import re

# --- 1. 기본 설정 ---
class Config:
    """스크립트의 경로 및 기본 설정을 관리합니다."""
    SAVE_PATH = 'C:/vscode_py/_save/dacon/kt/'
    OUTPUT_PREFIX = "sentence_classifier_prompt_final_no_markdown"
    CHAR_LIMIT = 4000

cfg = Config()
os.makedirs(cfg.SAVE_PATH, exist_ok=True)


# --- 2. 핵심 프롬프트 (성능 개선 및 마크다운 제거 버전) ---
CORE_SYSTEM_PROMPT = """
### 역할: 문장 분류기
당신은 주어진 한국어 문장을 4가지 속성으로 정확하게 분류하는 AI입니다.

### 핵심 판단 원칙
1.  독립적 분석: 각 속성(유형, 극성, 시제, 확실성)을 개별적으로 판단한 후 최종 결과를 조합합니다.
2.  키워드 우선: 규칙에 명시된 키워드(예: '못', '같다', '것이다')를 판단의 최우선 근거로 삼습니다.
3.  일반성 추론: 두 가지 이상의 속성이 가능할 경우, 더 보편적이거나 문맥상 자연스러운 쪽을 선택합니다.

### 분류 규칙
* 유형: 사실형(사실 전달), 추론형(원인/결과 분석), 대화형(질문/명령), 예측형(미래 전망)
* 극성: 긍정, 부정('안','못','없다'), 미정(질문/모호)
* 시제: 과거('했다'), 현재(진행/일반적 사실), 미래('할 것이다')
* 확실성: 확실, 불확실('같다','수 있다','보인다')

### 핵심 예시
-   입력: '길이 젖어있는 것을 보니 밤새 비가 온 것 같다.'
-   출력: `추론형,긍정,과거,불확실`
-   입력: '내일 프로젝트를 끝낼 수 있을까요?'
-   출력: `대화형,미정,미래,확실`
-   입력: '이 약은 부작용이 없습니다.'
-   출력: `사실형,부정,현재,확실`

### 출력 형식 (매우 중요)
-   오직 `유형,극성,시제,확실성` 순서로만 출력합니다.
-   쉼표(,) 외의 공백이나 다른 문자는 절대 포함하지 마십시오.
"""

def assemble_prompt(core_prompt: str, max_chars: int) -> str:
    """프롬프트의 길이를 최종 확인하고 반환합니다."""
    prompt = core_prompt.strip()
    if len(prompt) > max_chars:
        return prompt[:max_chars]
    return prompt


# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    try:
        print(">> 성능이 개선된 프롬프트를 생성합니다...")

        final_prompt = assemble_prompt(
            CORE_SYSTEM_PROMPT,
            cfg.CHAR_LIMIT
        )

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_filename = f"{cfg.OUTPUT_PREFIX}_{timestamp}.txt"
        out_path = Path(cfg.SAVE_PATH) / out_filename

        out_path.write_text(final_prompt, encoding="utf-8")

        print(f"✅ 프롬프트 생성이 완료되었습니다!")
        print(f" -> 저장 위치: {out_path}")
        print("\n이제 이 파일을 열어 내용을 복사하여 사용하시면 됩니다.")

    except Exception as e:
        print(f"코드 실행 중 오류가 발생했습니다: {e}")

