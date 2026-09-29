import html
import json
import os
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
# 대상 언어: 언어를 추가하려면 이 딕셔너리만 수정한다
# space: 긴 글을 나눠 번역한 뒤 문장 조각을 다시 이어 붙일 때 쓰는 구분자
LANGUAGES = {
    "en": {"badge": "EN", "name": "English", "space": " "},
    "ja": {"badge": "JA", "name": "日本語", "space": ""},
    "vi": {"badge": "VI", "name": "Tiếng Việt", "space": " "},
}
MAX_CHARS = 5000
# 한 번에 너무 긴 글을 보내면 모델이 문장을 빠뜨리므로 이 길이 단위로 나눠 병렬 번역한다
CHUNK_CHARS = 800
MAX_WORKERS = 6
# 원문보다 문장 수가 이 비율 미만이면 누락 가능성을 경고한다 (문장을 합치거나 나누는 정상적인 번역을 고려한 여유)
OMISSION_RATIO = 0.85
OMISSION_MIN_SENTENCES = 5
# 문장 끝: 서양식 문장부호는 뒤에 공백이 있을 때만(3.5 같은 숫자 제외), 전각 문장부호는 공백 없이도 끝으로 본다
SENTENCE = re.compile(r".+?(?:[.!?]+(?:\s+|$)|[。！？]+\s*|$)")
NUMBER_LABEL = re.compile(r"\s*\d+[.)]\s*")

SYSTEM_PROMPT = (
    "You are a professional translator. Detect the language of the user's text "
    "and translate it into "
    + ", ".join(f"{lang['name']} ({code})" for code, lang in LANGUAGES.items())
    + ". Preserve the original meaning, tone, and formality. "
    "Translate every sentence without omitting, summarizing, or merging anything, "
    "and keep the original line breaks, numbering, and formatting. "
    "Use natural equivalents for culture-specific terms instead of word-for-word translations. "
    "Each value must be a single string containing the complete translation. "
    "Do not add explanations."
)
RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "translations",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {code: {"type": "string"} for code in LANGUAGES},
            "required": list(LANGUAGES),
            "additionalProperties": False,
        },
    },
}


class TranslationParseError(Exception):
    """모델 응답을 번역 결과로 해석할 수 없을 때 발생한다."""


@st.cache_resource
def get_client(api_key: str) -> OpenAI:
    # 재시도는 1회로 제한해 응답이 없을 때 최대 대기 시간을 약 90초로 줄인다
    return OpenAI(api_key=api_key, timeout=45, max_retries=1)


def split_sentences(line: str) -> list[str]:
    """한 줄을 문장으로 나눈다. 이어 붙이면 원래 줄이 된다."""
    sentences = []
    for piece in SENTENCE.findall(line):
        # "36. "처럼 번호만 있는 조각은 다음 문장에 붙인다. 번호만 조각 끝에 남으면 모델이 번호를 빠뜨린다
        if sentences and NUMBER_LABEL.fullmatch(sentences[-1]):
            sentences[-1] += piece
        else:
            sentences.append(piece)
    return sentences


def split_line(line: str) -> list[str]:
    """CHUNK_CHARS보다 긴 줄을 문장 단위로, 문장도 너무 길면 공백·글자 단위로 자른다. 조각을 이으면 원래 줄이 된다."""
    if len(line) <= CHUNK_CHARS:
        return [line]
    pieces = []
    for sentence in split_sentences(line):
        if len(sentence) <= CHUNK_CHARS:
            pieces.append(sentence)
        else:
            pieces += textwrap.wrap(
                sentence, CHUNK_CHARS, expand_tabs=False, replace_whitespace=False, drop_whitespace=False
            )
    return pieces


def split_into_chunks(text: str) -> tuple[list[str], list[str]]:
    """텍스트를 CHUNK_CHARS 이하 조각으로 나눈다.

    조각 사이 경계가 줄바꿈이면 "\\n", 줄 중간이면 " "를 separators에 담아 돌려준다.
    """
    units = []  # (조각, 조각 뒤 경계)
    for line in text.split("\n"):
        pieces = split_line(line)
        units += [(piece, " ") for piece in pieces[:-1]] + [(pieces[-1], "\n")]

    chunks, separators, buf, last_sep = [], [], "", ""
    for piece, sep in units:
        # 줄 중간 경계의 공백은 앞 조각에 이미 들어 있으므로 줄바꿈만 다시 붙인다
        glue = "\n" if last_sep == "\n" else ""
        if buf and len(buf) + len(glue) + len(piece) > CHUNK_CHARS:
            chunks.append(buf)
            separators.append(last_sep)
            buf = piece
        else:
            buf = buf + glue + piece if buf else piece
        last_sep = sep
    chunks.append(buf)
    return chunks, separators


def translate_chunk(client: OpenAI, model: str, text: str) -> dict[str, str]:
    """조각 하나를 API 1회 호출로 LANGUAGES의 모든 언어로 번역한다."""
    response = client.chat.completions.create(
        model=model,
        response_format=RESPONSE_FORMAT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    try:
        data = json.loads(response.choices[0].message.content or "")
    except json.JSONDecodeError as e:
        raise TranslationParseError from e
    if not isinstance(data, dict):
        raise TranslationParseError
    # 문자열이 아닌 값(목록 등)은 화면에 파이썬 표기로 새지 않도록 빈 결과로 취급한다
    return {code: data[code].strip() if isinstance(data.get(code), str) else "" for code in LANGUAGES}


def translate(client: OpenAI, model: str, text: str) -> dict[str, str]:
    """텍스트를 번역한다. CHUNK_CHARS 이하는 1회 호출, 그보다 길면 나눠서 병렬 호출한다."""
    chunks, separators = split_into_chunks(text)
    if len(chunks) == 1:
        return translate_chunk(client, model, text)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        parts = list(pool.map(lambda chunk: translate_chunk(client, model, chunk), chunks))

    result = {}
    for code, lang in LANGUAGES.items():
        # 어느 조각이든 비어 있으면 전체를 비워서 '일부 누락' 경고가 뜨게 한다
        if not all(part[code] for part in parts):
            result[code] = ""
            continue
        joined = parts[0][code]
        for sep, part in zip(separators, parts[1:]):
            joined += ("\n" if sep == "\n" else lang["space"]) + part[code]
        result[code] = joined
    return result


def count_sentences(text: str) -> int:
    return sum(len(split_sentences(line)) for line in text.split("\n") if line.strip())


def find_omissions(source: str, translations: dict[str, str]) -> dict[str, int]:
    """원문보다 문장 수가 눈에 띄게 적은 언어와 그 문장 수를 돌려준다. 모델이 문장을 빠뜨렸는지 가늠하는 용도다."""
    expected = count_sentences(source)
    if expected < OMISSION_MIN_SENTENCES:
        return {}
    counts = {code: count_sentences(text) for code, text in translations.items() if text}
    return {code: n for code, n in counts.items() if n < expected * OMISSION_RATIO}


def error_message(e: Exception) -> str | None:
    """PRD 6장 오류 처리 정책에 따른 사용자용 메시지. 처리 대상이 아니면 None."""
    # APIError가 나머지 OpenAI 예외의 부모 클래스이므로 순서가 중요하다
    if isinstance(e, AuthenticationError):
        return "API 키가 올바르지 않습니다. .env 파일의 OPENAI_API_KEY를 확인하세요."
    if isinstance(e, RateLimitError):
        return "요청 한도를 초과했습니다. 잠시 후 다시 시도하세요."
    if isinstance(e, BadRequestError):
        return "요청이 거부되었습니다. OPENAI_MODEL이 JSON 스키마 응답을 지원하는 모델인지 확인하세요."
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        return "OpenAI 서버에 연결할 수 없습니다. 네트워크 상태를 확인하고 다시 시도하세요."
    if isinstance(e, APIStatusError):
        return f"API 오류가 발생했습니다. (코드 {e.status_code}) 잠시 후 다시 시도하세요."
    if isinstance(e, APIError):
        return "API 오류가 발생했습니다. 잠시 후 다시 시도하세요."
    if isinstance(e, TranslationParseError):
        return "번역 결과를 해석하지 못했습니다. 다시 시도하세요."
    return None


def escape_markdown(text: str) -> str:
    # 캡션은 "원문: "으로 시작하므로 줄 앞에서만 의미가 있는 #, -, + 등은 이스케이프하지 않는다
    # (URL 안의 . ) 을 이스케이프하면 자동 링크 처리 때문에 역슬래시가 그대로 보인다)
    return re.sub(r"([\\`*_\[\]<>|~$])", r"\\\1", text)


def request_translation() -> None:
    """번역하기 버튼 콜백: 입력을 검사하고 번역할 텍스트를 pending에 올린다."""
    text = st.session_state.input_text.strip()
    if not text:
        st.session_state.notice = ("warning", "번역할 텍스트를 입력하세요.")
    elif len(text) > MAX_CHARS:
        st.session_state.notice = ("warning", f"{MAX_CHARS:,}자 이하로 입력하세요. (현재 {len(text):,}자)")
    else:
        st.session_state.notice = None
        st.session_state.pending = text


# 테마(색·글꼴·모서리)는 .streamlit/config.toml에서 정하고, 여기에는 레이아웃 보정만 둔다
CUSTOM_CSS = """
<style>
[data-testid="stMainBlockContainer"] { max-width: 1100px; padding-top: 2.5rem; }
.app-subtitle { opacity: 0.7; margin: -0.75rem 0 1.5rem; }
.lang-label { font-weight: 600; }
.lang-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.1rem 0.45rem;
    margin-right: 0.45rem;
    border-radius: 0.4rem;
    /* 글자색은 테마 본문색을 따르게 해 라이트·다크 모드 모두에서 읽히게 한다 */
    background: rgba(99, 102, 241, 0.2);
    color: inherit;
}
.empty-state {
    border: 1px dashed rgba(128, 128, 128, 0.4);
    border-radius: 0.75rem;
    padding: 2rem 1rem;
    text-align: center;
    opacity: 0.65;
}
.app-footer { opacity: 0.6; font-size: 0.85rem; text-align: center; }
</style>
"""

st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("🌐 AI 번역기")
st.markdown(
    '<p class="app-subtitle">텍스트를 입력하면 영어 · 일본어 · 베트남어로 한 번에 번역합니다.</p>',
    unsafe_allow_html=True,
)

api_key = os.getenv("OPENAI_API_KEY", "").strip()
model = os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_MODEL

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
    st.stop()

pending = st.session_state.get("pending")

# st.form으로 묶어 Ctrl+Enter로도 제출되게 한다
with st.form("translate_form"):
    # max_chars를 쓰면 한도를 넘는 붙여넣기가 통째로 무시되므로, 길이는 콜백에서 검사한다
    st.text_area(
        "번역할 텍스트",
        key="input_text",
        height=150,
        placeholder="예: 내일 회의는 오후 3시에 시작합니다.",
        help=f"최대 {MAX_CHARS:,}자까지 번역할 수 있어요.",
    )
    # 번역 중에는 버튼을 비활성화해 중복 호출을 막는다
    st.form_submit_button(
        "번역 중..." if pending else "번역하기",
        type="primary",
        icon=":material/translate:",
        on_click=request_translation,
        disabled=pending is not None,
    )

if pending:
    if len(pending) <= CHUNK_CHARS:
        spinner_text = "번역 중..."
    else:
        spinner_text = "긴 글을 나눠서 번역하고 있어요. 30초 이상 걸릴 수 있어요..."
    try:
        with st.spinner(spinner_text):
            st.session_state.result = {
                "source": pending,
                "translations": translate(get_client(api_key), model, pending),
            }
    except Exception as e:
        message = error_message(e)
        if message is None:
            raise
        # 이전 결과가 새 번역으로 오해받지 않도록 오류가 나면 결과를 지운다
        st.session_state.result = None
        st.session_state.notice = ("error", message)
    finally:
        st.session_state.pending = None
    # 버튼을 다시 활성화하기 위해 한 번 더 그린다
    st.rerun()

notice = st.session_state.get("notice")
if notice:
    kind, message = notice
    (st.error if kind == "error" else st.warning)(message)

# 결과를 session_state에 두어 화면이 다시 그려져도 유지한다
result = st.session_state.get("result")
st.subheader("번역 결과")
if result:
    translations = result["translations"]
    source = result["source"]
    preview = source if len(source) <= 80 else source[:80] + "…"
    st.caption(f"원문: {escape_markdown(preview)}")
    if not all(translations.values()):
        st.warning("일부 언어의 번역 결과가 비어 있어요. 다시 시도해 주세요.")
    omissions = find_omissions(source, translations)
    if omissions:
        detail = ", ".join(f"{LANGUAGES[code]['name']} {n}문장" for code, n in omissions.items())
        st.warning(
            f"번역에서 일부 문장이 빠졌을 수 있어요. (원문 {count_sentences(source)}문장 → {detail}) "
            "다시 시도하거나 글을 나눠서 번역해 보세요."
        )
    # 좁은 화면에서는 Streamlit이 열을 자동으로 세로로 쌓는다
    for column, (code, lang) in zip(st.columns(len(LANGUAGES)), LANGUAGES.items()):
        with column, st.container(border=True):
            st.markdown(
                f'<span class="lang-label"><span class="lang-badge">{lang["badge"]}</span>{lang["name"]}</span>',
                unsafe_allow_html=True,
            )
            # st.code는 우측 상단에 복사 버튼을 제공한다 (F-9)
            st.code(translations[code] or "(결과 없음)", language=None, wrap_lines=True)
else:
    st.markdown(
        '<div class="empty-state">텍스트를 입력하고 <b>번역하기</b>를 눌러주세요.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    f'<p class="app-footer">모델: <code>{html.escape(model)}</code> · 입력한 텍스트는 OpenAI 서버로 전송됩니다.</p>',
    unsafe_allow_html=True,
)
