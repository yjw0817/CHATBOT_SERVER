# Argos RAG 전처리 시스템 — 세션 핸드오프 프롬프트

> 이 문서를 다음 세션 시작 시 전달하면 바로 이어서 작업할 수 있습니다.

---

## 프로젝트 개요

**Argos**는 아파트 커뮤니티 운영 문서(DOCX)를 업로드하여, AI가 RAG 지식베이스용 구조화 데이터로 변환하고, 챗봇이 검색·답변에 활용하는 시스템입니다.

- **스택**: FastAPI + SQLite + Jinja2 템플릿 + Gemini LLM
- **경로**: `d:\Projects\CHATBOT_SERVER\argos\`
- **실행**: `.\venv\Scripts\python -m uvicorn main:app --reload --port 8000`

---

## ✅ 완료된 작업

### 1. Manualize 버튼 상태 변경
- `routers/ui.py`: `has_manual` 플래그 추가 (manual_sections 테이블 조회)
- `templates/upload.html`: 매뉴얼 있으면 "📖 매뉴얼 보기", 없으면 "📝 Manualize" 표시

### 2. 에디터 UX 개선
- textarea 자동 확장 (`autoResize`, `editor-textarea` 클래스)
- AI 제안 전후 비교 모달 (`compare-modal`) — 좌우 비교 + 수정 가능
- 비교 모달 높이 350px / max 60vh로 확대

### 3. 섹션 저장 버그 수정
- 섹션 이름에 특수문자(`/`, 공백)가 있으면 `getElementById` 실패 → 인덱스 기반 ID + `data-section-name` 속성으로 전환
- "적용하기" 클릭 시 자동 DB 저장 (`applyCompare → saveEditor`)

### 4. RAG 전처리 목적 프롬프트 재작성

#### REFINE_PROMPT (다듬기 + 맥락 보강)
- 위치: `routers/api.py` — `REFINE_PROMPT` 상수
- 역할: `refine_text` 엔드포인트에서 사용
- task별 지침:
  - `refine` (② RAG 최적화): 대명사→구체명칭, 주어복원, 키워드명시, bullet point
  - `fill` (① 맥락 보강): 청크 독립성 확보, 암묵 조건 명시, 약어 풀기, Q&A 추가
- 원본 raw_text(4000자)를 함께 전달하여 hallucination 방지
- 필수규칙: 원본에 없는 정보 금지, [확인 필요] 라벨

#### MANUALIZE_PROMPT V2 (11.md 기반)
- 위치: `routers/api.py` — `MANUALIZE_PROMPT` 상수
- 출력 스키마: `doc_title`, `doc_type`, `summary`, `sections[]` (각 section에 `section_id`, `tags`, `content[]`(rule 배열: `rule_id`, `title`, `bullets`, `structured`, `source_quotes`, `issues`)), `clarification_questions`, `pii_handling`, `change_summary`
- 파서: V2 content 배열 → plain text로 flatten → `manual_sections.section_text`에 저장
- V1 하위호환 유지

### 5. UI 워크플로우 정리
- 버튼 순서: **① 맥락 보강** → **② RAG 최적화** (번호 표시)
- ② 버튼은 ①이 완료될 때까지 disabled (opacity 0.5)
- ① 완료 시 ✅ 표시 + ② 활성화
- "다시 Manualize" 버튼은 에디터 모달 내부에만 배치 (클릭 시 에디터 닫고 진행)
- 혼동되는 "AI 재검증 (Gate)" 버튼 제거 (Gate는 테이블에서만)

---

## 📌 현재 아키텍처 (데이터 흐름)

```
DOCX 업로드 → Extract (raw_text 추출)
  → Manualize (V2 프롬프트 → JSON → flatten → manual_sections.section_text 저장)
    → 에디터에서 수동 편집 또는:
      ① 맥락 보강 (fill) → 비교 모달 → 적용 → section_text 덮어쓰기
      ② RAG 최적화 (refine) → 비교 모달 → 적용 → section_text 덮어쓰기
    → Gate (품질 검사: MISSING/AMBIGUOUS/CONFLICT/PII_RISK/API_NEEDED)
      → RED 이슈 0개 → RAG 데이터화 (Approve → Reindex → chunks 생성)
        → 챗봇 검색/답변에 활용
```

---

## ⚠️ 현재 알려진 이슈 / 개선 필요 사항

### 1. V2 구조화 데이터 소실
- **문제**: Manualize V2의 JSON 구조(section_id, tags, structured, source_quotes)가 flatten 시 소실됨
- **원인**: DB에 `section_text` (plain text)만 저장하므로 V2 구조 유지 불가
- **해결 방안**: `manual_sections` 테이블에 `section_json` 컬럼 추가하여 V2 원본 JSON도 보존. 에디터는 section_text 표시, RAG 청킹 시 structured 데이터 활용

### 2. 편집 히스토리 없음
- **문제**: fill/refine 적용 시 이전 section_text가 덮어쓰기되어 복구 불가
- **해결 방안**: `section_history` 테이블 또는 `manual_sections`에 `prev_text` 컬럼

### 3. fill 프롬프트의 한계
- **문제**: 현재 fill은 해당 섹션의 텍스트 + raw_text만 전달. 다른 섹션의 내용은 모름
- **해결 방안**: 같은 문서의 다른 섹션 텍스트도 컨텍스트로 전달 (토큰 한도 내에서)

### 4. 팝업 스타일링 (7.md)
- Gate/Extract/Manualize 결과 팝업이 plain text로 표시됨
- 카드/뱃지/리스트 형태로 예쁘게 렌더링 필요

### 5. 청킹 고도화
- 현재: 단순 500자 고정 청킹 + 100자 오버랩
- 개선: 섹션/rule 단위 의미 기반 청킹, tags/metadata 포함

---

## 🎯 다음 세션 작업 우선순위 (제안)

### P1 — 즉시 필요
1. **V2 JSON 원본 보존**: `section_json` 컬럼 추가 + Manualize 시 저장
2. **fill 시 다른 섹션 컨텍스트 전달**: 같은 문서의 전체 섹션 요약을 fill 프롬프트에 포함
3. **편집 히스토리**: 최소 1단계 undo 가능하도록 이전 텍스트 보관

### P2 — 품질 향상
4. **팝업 스타일링** (7.md 참고): Gate/Manualize 결과를 카드 UI로 표시
5. **의미 기반 청킹**: rule 단위로 청크 생성, tags를 메타데이터로 저장
6. **PII 마스킹 결과 표시**: Manualize 완료 후 PII 탐지 결과를 UI에 경고로 표시

### P3 — 추가 기능
7. **API 스펙 추출 UI 개선**: 추출된 API 목록을 별도 탭/카드로 표시
8. **일괄 처리**: 전체 섹션에 대해 ① 맥락 보강 → ② RAG 최적화를 한 번에 실행하는 "자동 전처리" 버튼

---

## 주요 파일 위치

| 파일 | 역할 |
|------|------|
| `argos/routers/api.py` | 모든 API 엔드포인트 + LLM 프롬프트 (MANUALIZE_PROMPT, REFINE_PROMPT, QUALITY_GATE_PROMPT) |
| `argos/routers/ui.py` | UI 라우터 (템플릿 렌더링 + has_manual 플래그) |
| `argos/templates/upload.html` | 문서 업로드/관리 페이지 (에디터 모달, 비교 모달 포함) |
| `argos/database.py` | SQLite 스키마 (documents, manual_sections, qa_issues, chunks 등) |
| `argos/llm_client.py` | LLM 클라이언트 (Gemini/OpenAI/Claude/Ollama 지원) |
| `11.md` | Manualize V2 프롬프트 원본 |
| `10.md` | AI 문서 처리 지침 원본 |
