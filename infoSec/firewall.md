## 방화벽 유형 (Firewall Types)

#### 1) 패킷 필터링 방화벽 (Packet Filtering)

- 가장 기본. IP, Port, 프로토콜 기반으로 허용/차단

- 빠름

- 내용은 못 봄(Layer 3/4)

#### 2) 상태기반 방화벽 (Stateful Inspection)

- 연결의 “상태(State)”를 기억하며 판단
- 패킷 흐름을 추적해서 더 정확함
- 일반 기업에서 가장 많이 사용

#### 3) 프록시 방화벽 (Application Firewall)

- 내용(application layer) 분석
- HTTP, FTP, 이메일 같은 애플리케이션 레벨 검사
- 제일 안전하지만 느림