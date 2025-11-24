## IDS vs IPS (침입 탐지/방지 시스템)

#### 1) IDS (Intrusion Detection System)
- 공격을 “탐지”만 한다.
- 네트워크/시스템을 모니터링
- 이상한 패턴, 공격 흔적을 알림
- 하지만 차단은 하지 않음
- ex) 지금 SQL Injection 시도 들어왔음!” → 관리자에게 알림

#### 2) IPS (Intrusion Prevention System)
- 공격을 탐지하고 즉시 차단
- IDS 기능 + 차단 기능
- 실시간 대응
- 하지만 너무 민감하면 정상 트래픽까지 막는 오탐 이슈
- ex) SQL Injection 패턴 감지 → 요청 자체를 즉시 차단