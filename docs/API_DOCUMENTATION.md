# Ollama Gateway API 문서

## Remote 서버 접속 방식 (2가지)

Argos 시스템에서 Remote Ollama 서버에 접속하는 방식은 두 가지가 있다.

### 방식 A: Gateway 서버 경유 (포트 8000)

| 항목 | 값 |
|------|-----|
| URL | `http://<서버IP>:8000` |
| 인증 | JWT Bearer Token (`LLM_API_KEY`) |
| 프로토콜 | OpenAI-compatible (`/v1/chat/completions`) |
| 모델 목록 | `/api/tags` **미지원** (Gateway 자체 API만 제공) |

- Gateway가 세션 분리, Rate Limiting, 동시 처리 제한 등을 담당
- 여러 사용자가 공유하는 환경에 적합
- **제약:** Ollama 네이티브 API(`/api/tags`, `/api/embeddings`)를 직접 호출할 수 없음
- Argos의 동적 모델 목록 조회(`GET /api/llm/models`)가 동작하지 않음

### 방식 B: Ollama 직접 접속 (포트 11434)

| 항목 | 값 |
|------|-----|
| URL | `http://<서버IP>:11434` |
| 인증 | 불필요 (또는 Bearer Token) |
| 프로토콜 | Ollama 네이티브 + OpenAI-compatible 모두 지원 |
| 모델 목록 | `/api/tags` **지원** |

- Ollama 서버에 직접 연결하므로 모든 API 사용 가능
- `/api/tags` → 동적 모델 목록 조회
- `/api/embeddings` → 임베딩 생성 (RAG)
- `/v1/chat/completions` → LLM 채팅
- **제약:** 세션 분리/Rate Limiting 없음 (단일 사용자 또는 내부 서비스용)

### .env 설정 예시

```env
# 방식 A: Gateway 경유
LLM_REMOTE_URL=http://192.168.0.28:8000

# 방식 B: Ollama 직접 (현재 사용 중)
LLM_REMOTE_URL=http://192.168.0.28:11434
```

### Argos 시스템 권장

- **Argos 백엔드(매뉴얼화/RAG/채팅):** 방식 B (직접 접속) 권장 — 임베딩 + 모델 목록 조회 필요
- **일반 사용자 채팅 클라이언트:** 방식 A (Gateway) 권장 — 세션 분리/Rate Limiting 필요

---

## 개요

사내 Ollama AI 중계 API 서버입니다. 여러 직원이 동시에 사용해도 대화가 섞이지 않도록 설계되었습니다.

### 주요 특징

- **세션 분리**: (user_id, conversation_id) 기반 완전한 대화 분리
- **JWT 인증**: Bearer 토큰 기반 사용자 인증
- **Rate Limiting**: 사용자별 분당 10회 요청 제한
- **동시 처리 제한**: Ollama 동시 요청 2개로 제한
- **컨텍스트 관리**: 최근 30개 메시지만 AI에 전달

---

## 서버 정보

| 항목 | 값 |
|------|-----|
| 기본 URL | `http://<서버IP>:8000` |
| 인증 방식 | JWT Bearer Token |
| 콘텐츠 타입 | `application/json` |

---

## 인증

### 인증 흐름

```
1. POST /api/auth/login → access_token 발급
2. 이후 모든 요청에 헤더 추가: Authorization: Bearer {access_token}
```

### 토큰 사용법

```bash
curl -X GET "http://서버IP:8000/api/conversations" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..."
```

---

## API 엔드포인트

### 1. 헬스체크

서버 상태를 확인합니다.

```
GET /health
```

**요청**: 인증 불필요

**응답 (200)**:
```json
{
  "status": "healthy"
}
```

---

### 2. 회원가입

새 사용자를 등록합니다.

```
POST /api/auth/register
```

**요청**:
```json
{
  "username": "홍길동",
  "password": "비밀번호123"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| username | string | ✅ | 사용자명 (3-100자) |
| password | string | ✅ | 비밀번호 (4-100자) |

**응답 (201)**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "홍길동",
  "is_active": true
}
```

**오류**:
| 코드 | 설명 |
|------|------|
| 400 | 이미 존재하는 사용자명 |

---

### 3. 로그인

JWT 토큰을 발급받습니다.

```
POST /api/auth/login
```

**요청**:
```json
{
  "username": "홍길동",
  "password": "비밀번호123"
}
```

**응답 (200)**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**오류**:
| 코드 | 설명 |
|------|------|
| 401 | 사용자명 또는 비밀번호 오류 |
| 403 | 비활성화된 사용자 |

---

### 4. 내 정보 조회

현재 로그인한 사용자 정보를 조회합니다.

```
GET /api/auth/me
```

**요청**: 인증 필요

**응답 (200)**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "홍길동",
  "is_active": true
}
```

---

### 5. 대화 생성

새 대화를 생성합니다.

```
POST /api/conversations
```

**요청**: 인증 필요
```json
{
  "title": "업무 문의"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| title | string | ❌ | 대화 제목 |

**응답 (201)**:
```json
{
  "id": "74b8cb98-dd0b-466c-afe8-668127bc9f15",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "업무 문의",
  "created_at": "2024-01-15T09:30:00",
  "updated_at": "2024-01-15T09:30:00"
}
```

---

### 6. 대화 목록 조회

사용자의 대화 목록을 조회합니다.

```
GET /api/conversations
```

**요청**: 인증 필요

**응답 (200)**:
```json
{
  "conversations": [
    {
      "id": "74b8cb98-dd0b-466c-afe8-668127bc9f15",
      "user_id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "업무 문의",
      "created_at": "2024-01-15T09:30:00",
      "updated_at": "2024-01-15T10:00:00"
    }
  ]
}
```

---

### 7. 메시지 목록 조회

대화의 전체 메시지를 조회합니다.

```
GET /api/conversations/{conversation_id}/messages
```

**요청**: 인증 필요

**응답 (200)**:
```json
{
  "messages": [
    {
      "id": "msg-001",
      "conversation_id": "74b8cb98-dd0b-466c-afe8-668127bc9f15",
      "role": "user",
      "content": "안녕하세요",
      "created_at": "2024-01-15T09:30:00",
      "metadata": null
    },
    {
      "id": "msg-002",
      "conversation_id": "74b8cb98-dd0b-466c-afe8-668127bc9f15",
      "role": "assistant",
      "content": "안녕하세요! 무엇을 도와드릴까요?",
      "created_at": "2024-01-15T09:30:05",
      "metadata": {
        "model": "qwen3:30b",
        "tokens": 15,
        "latency_ms": 1200
      }
    }
  ]
}
```

---

### 8. 메시지 전송 ⭐ (핵심)

메시지를 전송하고 AI 응답을 받습니다.

```
POST /api/conversations/{conversation_id}/messages
```

**요청**: 인증 필요
```json
{
  "message": "오늘 회의 안건을 정리해줘"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| message | string | ✅ | 사용자 메시지 (최대 10,000자) |

**응답 (200)**:
```json
{
  "answer": "네, 회의 안건을 정리해 드리겠습니다...",
  "conversation_id": "74b8cb98-dd0b-466c-afe8-668127bc9f15",
  "message_id": "msg-003"
}
```

**오류**:
| 코드 | 설명 | 대응 |
|------|------|------|
| 400 | 메시지가 너무 김 | 10,000자 이하로 |
| 401 | 인증 필요 | 로그인 후 재시도 |
| 404 | 대화를 찾을 수 없음 | conversation_id 확인 |
| 429 | 요청 제한 초과 | Retry-After 헤더 확인 후 대기 |
| 503 | 서버 과부하 | 잠시 후 재시도 |

---

## 제한사항

| 항목 | 제한 | 설명 |
|------|------|------|
| Rate Limit | 10회/분 | 사용자별 분당 요청 수 |
| 동시 처리 | 2개 | Ollama 동시 요청 수 |
| 메시지 길이 | 10,000자 | 한 번에 보낼 수 있는 최대 길이 |
| 컨텍스트 | 30개 | AI에 전달되는 최근 메시지 수 |
| 토큰 유효기간 | 30분 | JWT 토큰 만료 시간 |

---

## 에러 응답 형식

모든 에러는 다음 형식으로 반환됩니다:

```json
{
  "detail": "에러 메시지"
}
```

### Rate Limit (429) 응답

```
HTTP/1.1 429 Too Many Requests
Retry-After: 45

{
  "detail": "요청이 너무 많습니다. 45초 후 다시 시도하세요."
}
```

**처리 방법**: `Retry-After` 헤더의 초만큼 대기 후 재시도

---

## 사용 예시

### Python

```python
from gateway_client import GatewayClient

client = GatewayClient("http://서버IP:8000")
client.login("홍길동", "비밀번호123")

conv_id = client.create_conversation("업무 문의")
answer = client.chat(conv_id, "오늘 회의 안건 정리해줘")
print(answer)
```

### curl

```bash
# 로그인
TOKEN=$(curl -s -X POST "http://서버IP:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "홍길동", "password": "비밀번호123"}' \
  | jq -r '.access_token')

# 대화 생성
CONV_ID=$(curl -s -X POST "http://서버IP:8000/api/conversations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "업무 문의"}' \
  | jq -r '.id')

# 메시지 전송
curl -X POST "http://서버IP:8000/api/conversations/$CONV_ID/messages" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "안녕하세요!"}'
```

### JavaScript (fetch)

```javascript
// 로그인
const loginRes = await fetch('http://서버IP:8000/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: '홍길동', password: '비밀번호123' })
});
const { access_token } = await loginRes.json();

// 메시지 전송
const chatRes = await fetch(`http://서버IP:8000/api/conversations/${convId}/messages`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ message: '안녕하세요!' })
});
const { answer } = await chatRes.json();
```

---

## Swagger UI

자동 생성된 API 문서를 확인하려면:

```
http://서버IP:8000/docs
```

---

## 문의

서버 관련 문의: 서버 관리자에게 연락
