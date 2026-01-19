## 토큰 탈취 (Token Theft)
```js
핵심 개념
인증 토큰(JWT·세션)을 훔치면 비밀번호 없이 로그인 가능
주요 탈취 경로
XSS로 토큰/쿠키 탈취
평문 통신(HTTP)
브라우저 저장소(localStorage) 노출
로그·URL에 토큰 출력
```
- 왜 위험한가?
```js
MFA 있어도 이미 발급된 토큰이면 우회 가능
만료 전까지 사용자로 행동 가능

방어 방법
✔ HTTPS
✔ HttpOnly 쿠키 사용
✔ XSS 차단
✔ 짧은 토큰 만료
✔ Refresh Token 분리·회전
```