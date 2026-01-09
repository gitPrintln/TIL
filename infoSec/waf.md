## WAF (Web Application Firewall)
```js
핵심 개념
웹 공격(SQL Injection, XSS 등)을 애플리케이션 앞단에서 막는 방화벽
WAF는 무엇을 막나?
SQL Injection
XSS
CSRF 패턴
비정상적인 HTTP 요청
-> 네트워크 방화벽이 못 막는 “웹 로직 공격” 전용
```
- 동작 위치
```js
[사용자] → [WAF] → [웹서버/WAS]

장점 / 한계
✔ 소스 수정 없이 방어 가능
✔ 운영 중에도 룰 추가 가능
❌ 비즈니스 로직 취약점은 못 막음
❌ 오탐 가능성
```