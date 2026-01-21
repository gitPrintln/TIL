## 입력값 검증 (Input Validation)
```js
핵심 개념
사용자가 보낸 값은 절대 믿지 않는다
왜 중요한가?
SQL Injection
XSS
Command Injection
→ 대부분 입력값 검증 실패에서 시작
```
```js
기본 원칙
화이트리스트 방식 (허용할 것만 허용)
길이 제한
타입 검증
서버 단 검증 필수 (클라이언트만 믿지 말 것)

예시
숫자만 받아야 하는데 문자열 허용 → 취약
ID에 <script> 허용 → 
```