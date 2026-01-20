## 보안 헤더 (Security Headers)
```js
핵심 개념
HTTP 응답 헤더로 브라우저의 위험한 동작을 미리 차단
대표 보안 헤더
CSP (Content-Security-Policy)
허용된 스크립트만 실행 → XSS 방어 핵심

X-Frame-Options
다른 사이트에 iframe으로 못 끼우게 → 클릭재킹 방어

X-Content-Type-Options: nosniff
MIME 타입 추측 차단

Strict-Transport-Security (HSTS)
무조건 HTTPS로만 접속
```
- 왜 중요한가?
- 코드 수정 없이 보안 강화 가능
- XSS·클릭재킹 같은 프론트 공격 1차 방어선