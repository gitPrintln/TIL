## CORS (Cross-Origin Resource Sharing)
```js
핵심 개념
다른 출처(origin)에서의 요청을 허용할지 말지 정하는 브라우저 보안 규칙

왜 중요한가?
잘못 설정하면
인증 정보 탈취
내부 API 외부 노출

흔한 실수
Access-Control-Allow-Origin: *
Allow-Credentials: true 같이 사용 ❌

기본 원칙
필요한 도메인만 허용
인증 포함 요청은 특히 엄격
프론트 문제가 아니라 보안 설정 문제
```