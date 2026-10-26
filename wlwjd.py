import os
from collections import Counter
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from kiwipiepy import Kiwi

# ---------------------------------------------------------
# 0. 한글 폰트 설정 (NanumGothic.ttf)
# ---------------------------------------------------------
FONT_PATH = "NanumGothic.ttf"
if os.path.exists(FONT_PATH):
    font_prop = fm.FontProperties(fname=FONT_PATH)
    plt.rc("font", family=font_prop.get_name())
    plt.rcParams["axes.unicode_minus"] = False
else:
    font_prop = None

# 감성 사전 정의 (10개 이상)
POSITIVE_WORDS = {
    "좋다", "기쁘다", "행복하다", "최고", "훌륭하다",
    "만족하다", "감사하다", "즐겁다", "희망", "사랑하다", "추천하다"
}
NEGATIVE_WORDS = {
    "나쁘다", "슬프다", "화나다", "실망하다", "최악",
    "어렵다", "부족하다", "불만", "아쉽다", "괴롭다", "짜증나다"
}


# ---------------------------------------------------------
# 1. 텍스트 전처리 및 명사 추출 함수
# ---------------------------------------------------------
def preprocess_nouns(text: str, user_words: list[str]) -> list[str]:
    """사용자 사전을 등록하고, 2글자 이상의 명사(NNG, NNP, NR)만 추출"""
    kiwi = Kiwi()
    # 사용자 정의 단어 등록
    for word in user_words:
        if word.strip():
            kiwi.add_user_word(word.strip(), "NNP")

    tokens = kiwi.tokenize(text)
    # 명사 품사 태그(NNG, NNP, NR) 필터링 및 2글자 이상 조건
    nouns = [
        token.form for token in tokens
        if token.tag.startswith("N") and len(token.form) >= 2
    ]
    return nouns


# ---------------------------------------------------------
# 2. 감성 분석 함수 (동사/형용사 기본형 변환 매칭)
# ---------------------------------------------------------
def analyze_sentiment(text: str, user_words: list[str]) -> tuple[list[str], list[str]]:
    """Kiwi 토큰 분석을 통해 명사 및 동사/형용사의 기본형(~다)을 추출하여 감성 매칭"""
    kiwi = Kiwi()
    for word in user_words:
        if word.strip():
            kiwi.add_user_word(word.strip(), "NNP")

    tokens = kiwi.tokenize(text)
    pos_matched = []
    neg_matched = []

    for token in tokens:
        # 동사(VV) 및 형용사(VA)는 기본형 형태(~다)로 복원
        if token.tag in ("VV", "VA"):
            lemma = token.form + "다"
        else:
            lemma = token.form

        if lemma in POSITIVE_WORDS:
            pos_matched.append(lemma)
        elif lemma in NEGATIVE_WORDS:
            neg_matched.append(lemma)

    return pos_matched, neg_matched


# ---------------------------------------------------------
# 3. 시각화 함수 (막대그래프 & 워드클라우드)
# ---------------------------------------------------------
def plot_bar_chart(freq_df: pd.DataFrame):
    """상위 빈도 단어 가로 막대그래프 시각화"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_df = freq_df.sort_values(by="빈도", ascending=True)
    ax.barh(sorted_df["단어"], sorted_df["빈도"], color="skyblue")
    ax.set_xlabel("빈도수", fontproperties=font_prop)
    ax.set_ylabel("단어", fontproperties=font_prop)
    if font_prop:
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_prop)
    st.pyplot(fig)


def plot_wordcloud(freq_dict: dict[str, int]):
    """단어 빈도 기반 워드클라우드 시각화"""
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def plot_sentiment_pie(pos_count: int, neg_count: int):
    """긍정/부정 비율 파이차트 시각화"""
    fig, ax = plt.subplots(figsize=(4, 4))
    sizes = [pos_count, neg_count]
    labels = ["긍정", "부정"]
    colors = ["#66b3ff", "#ff9999"]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontproperties": font_prop}
    )
    ax.axis("equal")
    st.pyplot(fig)


# ---------------------------------------------------------
# 4. Streamlit UI 메인 화면
# ---------------------------------------------------------
def main():
    st.title("텍스트 단어 빈도 및 감성 분석기")

    # [사용자 사전] 입력 인터페이스
    user_input = st.text_input(
        "사용자 사전 단어 등록 (쉼표로 구분)",
        placeholder="예: 인공지능, 워드클라우드"
    )
    user_words = [w.strip() for w in user_input.split(",") if w.strip()]

    # 1. 파일 업로드
    st.subheader("1. 파일 업로드")
    uploaded_file = st.file_uploader("텍스트 파일(.txt)을 선택하세요.", type=["txt"])

    if uploaded_file is not None:
        raw_text = uploaded_file.read().decode("utf-8")

        # 2. 파일 내용 보기
        st.subheader("2. 파일 내용")
        st.text_area("원본 텍스트 미리보기", raw_text, height=150)

        # 3. 전처리 결과
        st.subheader("3. 전처리 결과 (2글자 이상 명사)")
        nouns = preprocess_nouns(raw_text, user_words)
        st.write(f"추출된 총 명사 수: **{len(nouns)}개**")
        st.write(", ".join(nouns[:50]) + ("..." if len(nouns) > 50 else ""))

        # 4. 단어 빈도
        st.subheader("4. 단어 빈도")
        if nouns:
            noun_counts = Counter(nouns)
            max_limit = min(50, len(noun_counts))
            top_n = st.slider("표시할 상위 단어 개수", min_value=1, max_value=max_limit, value=min(10, max_limit))

            top_nouns = noun_counts.most_common(top_n)
            freq_df = pd.DataFrame(top_nouns, columns=["단어", "빈도"])
            st.dataframe(freq_df, use_container_width=True)

            # 5. 시각화 (탭 2개)
            st.subheader("5. 단어 빈도 시각화")
            tab1, tab2 = st.tabs(["가로 막대그래프", "워드클라우드"])
            with tab1:
                plot_bar_chart(freq_df)
            with tab2:
                plot_wordcloud(dict(top_nouns))
        else:
            st.info("조건에 맞는 2글자 이상의 명사가 없습니다.")

        # 6. 감성 분석
        st.subheader("6. 감성 분석")
        pos_list, neg_list = analyze_sentiment(raw_text, user_words)
        pos_cnt, neg_cnt = len(pos_list), len(neg_list)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**긍정 단어 ({pos_cnt}개)**")
            st.write(", ".join(pos_list) if pos_list else "매칭 단어 없음")
        with col2:
            st.markdown(f"**부정 단어 ({neg_cnt}개)**")
            st.write(", ".join(neg_list) if neg_list else "매칭 단어 없음")

        if pos_cnt + neg_cnt > 0:
            plot_sentiment_pie(pos_cnt, neg_cnt)
        else:
            st.info("사전과 일치하는 감성 단어가 텍스트 내에 존재하지 않습니다.")


if __name__ == "__main__":
    main()
