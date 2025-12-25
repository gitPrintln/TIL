## 세션 하이재킹 (Session Hijacking)
```js
핵심 개념
로그인 이후에 발급된 세션을 탈취해서 사용자로 가장하는 공격

어떻게 일어날까?
평문 통신에서 세션 ID 탈취
XSS로 쿠키 탈취
세션 ID 예측
공용 PC·공용 Wi-Fi
```
```js
왜 위험한가?
비밀번호 몰라도 로그인 상태 탈취 가능
관리자 세션 탈취 시 치명적
```
- 방어 방법
```js
✔ HTTPS 필수
✔ HttpOnly / Secure 쿠키
✔ 세션 재발급(Session Regeneration)
✔ 짧은 세션 만료
✔ XSS 방어
```