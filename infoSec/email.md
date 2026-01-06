## 이메일 보안 (Email Security)
```js
- 핵심 개념
이메일은 가장 흔한 공격 시작점
```
- 대표 공격
1) Phishing (피싱)
```js
정상 메일로 위장해
ID/PW 탈취
악성코드 유포
```
2) Spear Phishing

```js
특정 조직·사람을 노린 맞춤형 피싱
```
3) SMTP Spoofing
```js
발신자 주소를 위조한 메일
방어 기술
✔ SPF: 허용된 메일 서버만 발송
✔ DKIM: 메일 위·변조 검증
✔ DMARC: SPF+DKIM 정책 
```