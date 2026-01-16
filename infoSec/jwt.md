## JWT (JSON Web Token)
```js
핵심 개념
서버가 “너 누구고 뭐 할 수 있는지”를 토큰 하나에 담아 주는 방식

- JWT 구조 (3부분)
Header.Payload.Signature

Header: 알고리즘 정보
Payload: 사용자 정보, 권한(클레임)
Signature: 위변조 방지 서명
```
```js
장점 / 단점
서버 상태 저장 필요 없음 (Stateless)
확장성 좋음

탈취 시 위험
즉시 무효화 어려움

보안 포인트
HTTPS 필수
짧은 만료시간
Refresh Token 분리
민감정보 넣지 말 것
```