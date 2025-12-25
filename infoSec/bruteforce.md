## 암호 공격 기법 (Password Attacks)

#### 핵심 개념

- 암호를 직접 깨거나 추측해서 인증을 우회하는 공격
```js
1) Brute Force Attack

가능한 모든 조합을 무차별 대입
성공률 높음
시간 오래 걸림
짧은 비밀번호에 치명적
```
```js
2) Dictionary Attack

자주 쓰는 단어·패턴 목록 사용
Brute Force보다 빠름
단순한 비밀번호에 매우 취약
```
```js
3) Credential Stuffing

유출된 ID/PW 조합을 다른 서비스에 시도
재사용 비밀번호 계정 털림
자동화 공격 많음
방어 방법
```
```js
✔ 비밀번호 길이·복잡도
✔ 계정 잠금 정책
✔ MFA 적용
✔ 해시 + Salt
```