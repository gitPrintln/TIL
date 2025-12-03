## 스니핑(Sniffing) & 스푸핑(Spoofing)

#### 1) Sniffing (도청/패킷 캡처)
```js
네트워크에서 흐르는 데이터를 몰래 가로채서 보는 행위

특징

평문 전송 시 ID/PW 그대로 털릴 수 있음

ARP 스푸핑과 함께 자주 사용

Wireshark 같은 툴로 패킷 분석 가능

방어

✔ HTTPS 사용
✔ 스위치 포트 보안
✔ 암호화
```
#### 2) Spoofing (위장)
```js
나를 다른 존재로 속여 위조하는 공격

종류 예시

IP Spoofing (IP주소 위조)

MAC Spoofing

DNS Spoofing (잘못된 사이트로 유도)

ARP Spoofing → 스니핑과 연계 자주 됨

핵심 개념

"나는 서버인 척", "나는 게이트웨이인 척" → 속이고 트래픽 가로채기
```