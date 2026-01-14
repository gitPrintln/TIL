## OAuth 2.0 (인증·인가 프레임워크)

```js
핵심 개념
비밀번호를 주지 않고 “권한”만 위임하는 방식

- OAuth가 왜 필요할까?
외부 서비스 연동 시
→ ID/PW 직접 주면 위험
대신 토큰으로 접근 권한만 부여
```
- 기본 등장인물
```js
- Resource Owner: 사용자
- Client: 앱/서비스
- Authorization Server: 인증 서버
- Resource Server: API 서버
```
```js
- 핵심 포인트
Access Token으로 API 접근
Scope로 권한 범위 제한
토큰 만료 필수
```