## 인증 vs 인가 (Authentication vs Authorization)
```js
핵심 개념
인증(Authentication): 너 누구야?
인가(Authorization): 너 이거 해도 돼?

왜 자주 사고 나나?
로그인만 하면 모든 API 접근 가능
관리자/일반 사용자 구분 누락

보안 포인트
인증 후 반드시 인가 체크
URL·API 단위 권한 분리
프론트가 아니라 서버에서 강제

예시
로그인 O + 관리자 페이지 접근 ❌
→ 인가 체크 누락
```