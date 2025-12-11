## MFA (다중요소 인증, Multi-Factor Authentication)


#### 핵심 개념

- 두 가지 이상 인증 요소를 동시에 사용해서 보안 강화하는 방식

#### MFA를 구성하는 세 요소
```js
아래 중 둘 이상을 조합:
지식 요소 (Something you know)
비밀번호, PIN, 패턴
소유 요소 (Something you have)
OTP(구글 OTP), 스마트폰 인증, 카드키
생체 요소 (Something you are)
지문, 얼굴, 홍채
```
#### 왜 필요한가?
- 비밀번호만 있으면 유출될 위험이 크기 때문
- MFA를 적용하면 계정 탈취 난이도가 대폭 상승
```js
예시
로그인: ID/PW 입력 → 휴대폰 OTP 추가
금융앱: 지문 + PIN
업무 시스템: 패스워드 + 인증앱 푸시 승인
``` 