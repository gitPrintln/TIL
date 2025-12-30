## DNS 공격 (DNS Attacks)
```js
핵심 개념
도메인 이름 ↔ IP 변환 과정(DNS)을 악용하는 공격
```
- 대표 공격 유형
1) DNS Spoofing (Cache Poisoning)
- DNS 응답을 위조해서 가짜 IP로 유도
- 정상 사이트 → 피싱 사이트로 이동
- 사용자 인지 어려움
- 예) bank.com 접속했는데 가짜 은행 사이트로 연결

2) DNS Amplification (증폭 공격)
- 작은 요청으로 아주 큰 응답을 유도하는 DDoS
- UDP 기반
- 반사(Reflection) + 증폭(Amplification)
```js
방어 방법

✔ DNSSEC 적용
✔ 캐시 관리
✔ 트래픽 제한
✔ 최신 DNS 서버 사용
```