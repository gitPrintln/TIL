## QC Market & Trend

## 숫자로 보는 2025년 Quantum Industry
-> $12.6B Investment
-> $1B+ Revenue
-> 300+ Company

## 치우쳐져 있는 현황
- 금융, 약품, 물류/운송, 화학에너지와 물질

## 민간 자본의 양자 유입 본격화 - 연구 단계를 넘어 상업화 가능성에 대한 시장의 기대 반영

## Trend
- Hybrid Trend, AI-HPC
- Mckinsey "The most credible path to near-term value remains hybrid"
- OECD "A hybrid classical-quantum strategy is widely recognised as the most promising near-term approach"

## Hybrid Computer
- 서버, 제어장치, QPU 를 한 곳에 설치하고 저지연으로 연결
- 소프트웨어 스택(미들웨어)이 작업을 나눠주고 결과를 합침

## Hybrid Computing
- QPU는 GPU처럼 가속기: 양자에 맞는 계산만 QPU가, 나머지는 CPU, GPU rk
- 한 번에 끝나지 않고 결과를 주고받으며 반복해 답을 개선
-> Hybrid 컴퓨터라는 장비 위에서, Hybrid 컴퓨팅이라는 방식으로 계산

## Hybrid Workfolw
- Classical prepration(계산 작업 시작) -> 고전, 양자 자원 연결 후 정보를 양자 회로로 변환 -> 하위 문제를 고전으로 처리 후 하위 문제를 양자로 처리 -> 양자,고전 결과 결합 후 결과를 고전 비트로 변환 -> 계속 반복(오류 낮춤) -> 최종 분석 및 HPC dnjzmvmffh xhdgkq

## Hybrid Infrastructure
- 기존 데이터 구조 연산 층에 QPU 추가
- 고전 플랫폼 자리에 Quantum-HPC 미들웨어가 들어가, QPU와 CPU-GPU를 하나의 자원처럼 관리
- HPC-coupled QC Deployments : HPC - 양자 컴퓨터 도입 건수 증가

## 일본의 사례
- RIKEN(일본 이화학연구소) 산하 R-CSS(RIKEN 계산과학연구센터)의 RIKENxIBM
- AIST(산업기술총합연구소) 산하 G-QuAT ABCI 슈퍼 컴퓨팅센터의 ABCI-Q

## 독일의 사례
- 독일 율리히(JSC) JUNIQ
- 독일 LRZ(라이프니츠 슈퍼컴퓨팅센터, 뮌헨), Euro-Q-Exa

## 미국의 사례
- Quantinuum Helios x Oracle(OCI)
- Rigetti x HPE x 피츠버그 슈퍼컴퓨팅센터(PSC) - "TangleLab"

## 핀란드 vs 미국(어떻게 시작하느냐에 차이)
- 핀란드 CSC핀란드 CSC(국가 과학IT센터)의 LUMI x HELMI
- 미국 ORNL(오크리지 국립연구소) QCUP

## 우리나라
- KISTI x NVIDIA x IonQ x HPE 국가 슈퍼컴퓨터 센터
- 충북대 x IQM Spark
- 산업부 지원 - 기업향 양자 생태계 구축(양자융합 산업기반) - 연세대

## 공급과 수요
- 투자는 하드웨어로, 수요기업 예산은 활용으로 - 그 사이를 잇는 '활용 역량'이 다음 과제
- usecase 활용이 44%

## QC Ecosystem - players
- 최종수요 기업, 양자 솔루션, 서비스 제공기업, 양자 공급망(양자 하드웨어 제조 기업, 비양자 기술 제조기업), 투자자, 정부기관, 대학, 기타 양자기술 참여 기관

## 양자 컴퓨팅 스택
- top stack: 무엇을 풀 것인가
- Users, Applications, Algorithms
- middle stack: 소프트웨어와 하드웨어를 잇는 층
- Frameworks, Architecture, Control logic
- hardware stack: 실제로 계산하는 물리적 장치
- Control Plane, Quantum Plane

## Quantum company
- 100년 넘은 IT 기업이자, 양자 컴퓨팅 역사의 출발점부터 함께 해온 회사
- D-Wave
- google
- MS
- quantinuum
- NVIDIA
- aws
- xanadu
- IQM
- pasqal
- classiq
- q-ctrl
- strangeworks
- qbraid
- qunova computing
- multiverse computing

## Quantum Ecosystem - users
- 기대 효과: '불가능을 가능케' 하는 기술적 돌파구 마련, 미래를 위한 '탐색을 위한 탐색', 비용절감에 대한 기대 증가 기술적 호기심에서 -> 경제적 가치에 대한 기대로 확장
- 기대 도입 시기: 향후 2~5년 사이 상용화 예측

## Quantum Advantage
- Quantum Supremacy
- 검증(validation) : 결과가 맞다는 것을 엄밀하게 확인할 수 있어야함 -> 오차 범위제시, 고전으로 검산 가능한 문제, 변분 문제
- 분리(sepration) : 최선의 고전 방법보다 확실히 앞서야함 -> 효율, 비용, 정확도
-> Key: Quantum Advantage는 선언이 아니라 반증 가능한 과학적 가설 - 한 번의 이벤트가 아닌, 점점 탄탄해지는 시연의 연속

- quantum advantage
-> 1. 최선의 고전 방법과 비교했나? -> 2. 고전으로 흉내 낼 수 없는 규모인가? -> 3. 결과가 맞는지 검증할 수 있나? -> 4. 무엇이, 얼마나 나아졌나? -> 5. 재현 가능하고, 시간이 지나도 버티나?
-> 단순히 양자를 썼다 != 우위에 섰다. -> 5가지 질문을 통과하면 Quantum Advantage

-> 최종적으로 Business Value 개선 효과로 도입비용(QPU,인력,통합)을 넘어서는 경제적 이익

## Quantum Readiness
- 인식, 기회 탐색
- use case 발굴, 선정
- 역량, 기반 확보
- Hybrid PoC 실행
- 평가, 확장