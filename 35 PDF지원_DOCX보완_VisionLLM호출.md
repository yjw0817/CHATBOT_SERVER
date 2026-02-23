argos/routers/api.py의 텍스트 추출 로직(276-321줄)을 아래 요구사항에 맞게 수정해줘.

**현재 상태:**
- DOCX: paragraphs만 추출 (테이블/이미지 누락)
- TXT/MD: 그대로 읽기
- PDF: 미구현 ("[PDF extraction not implemented]" 반환 중)

**요구사항:**

**1. PDF 지원 추가 (PyMuPDF 사용)**
- 텍스트: 페이지별 page.get_text() 추출 후 합치기
- 이미지: 페이지별 이미지 추출 → Vision LLM 호출로 설명 텍스트 생성
- 이미지 설명은 [이미지 설명: ...] 형태로 본문에 삽입

**2. DOCX 보완**
- 테이블: 행/열 텍스트화하여 본문에 포함
- 이미지: word/media/ 내 이미지를 zipfile로 추출 → Vision LLM 호출로 설명 텍스트 생성
- 이미지 설명은 [이미지 설명: ...] 형태로 본문에 삽입

**3. Vision LLM 호출 스펙**
- 모델: qwen3-vl:235b-cloud
- Ollama API 엔드포인트: http://localhost:11434/api/chat
- 이미지는 base64 인코딩하여 전달
- 프롬프트: "이 이미지를 한국어로 자세히 설명해줘. UI 화면이라면 어떤 기능의 화면인지, 버튼/메뉴/입력 필드 등 구성 요소를 포함해서 설명해."

**4. 기타**
- Vision LLM 호출 실패 시 해당 이미지는 [이미지 설명 실패]로 처리하고 계속 진행
- 필요한 라이브러리: PyMuPDF(fitz), zipfile(표준 라이브러리)
- 기존 TXT/MD 처리 로직은 그대로 유지

---