## OSI 7계층 (OSI 7 Layers)

#### 핵심 개념
네트워크 통신을 7단계로 나눈 표준 모델
→ 보안 장비·공격 위치 설명할 때 필수

- 7계층 한 줄씩

1️⃣ 물리층 (Physical)
```js
케이블, 전기 신호
도청, 케이블 절단
```
2️⃣ 데이터링크층 (Data Link)
```js
MAC 주소, 스위치
ARP Spoofing
```
3️⃣ 네트워크층 (Network)
```js
IP, 라우팅
IP Spoofing
```
4️⃣ 전송층 (Transport)
```js
TCP/UDP, 포트
SYN Flood
```
5️⃣ 세션층 (Session)
```js
세션 관리
세션 하이재킹
```
6️⃣ 표현층 (Presentation)
```js
암호화, 인코딩
SSL/TLS
```
7️⃣ 응용층 (Application)
```js
HTTP, FTP
SQL Injection, XSS
```