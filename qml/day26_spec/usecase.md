## Quantum Simulation
- From hamiltonian to discovery
- Hamiltonian 전자들이 배치가 되어 있을 때 어떤식으로 정의 되어 있는지 규칙 같은 것

=> 화학 소재 바이오의 전자 구조 문제를 양자 알고리즘으로 계산하고 에너지 상태 정보를 활용

- QPE: 위상 추정 기반 에너지 계산
- VQE: 양자회로 측정 + 고전 파라미터 최적화
- SQD: 양자 측정 샘플 + 고전 부분 공간 대각화

## VQE: Variational Quantum Eigensolver
- 상대적으로 얕은 회로를 반복 실행해 바닥상태 에너지를 근사하는 Hybrid Variational Algorithm
- Hamlitoian 설정 - 큐비트 연산자로 변환 - 양자 파라미터화된 회로(Ansatz)로 trial 상태 생성(후보 생성) - energy measurement(측정) - 측정에너지를 바탕으로 파라미터 업데이트, Ground State 근사치 획득 - 반복수행

## Drug Binding Energy Simulation
- Mitsubishi Chemical

## ADAPT-VQE
- ORP Catalyst Simulation

## SQD
- 얕은 회로는 유지하면서 측정샘플을 더 효과적으로 활용하는 법
- 양자는 중요한 전자 배치를 샘플링하고 고전 컴퓨터가 subspace에서 정확히 대각화하는 샘플 기반 알고리즘
- 1. 후보 뽑기(양자 컴퓨터)
- 2. 바로잡기(일반 컴퓨터)
- 3. 후보 추리기(일반 컴퓨터)
- 4. 정밀 계산(슈퍼 컴퓨터)

## carbon Capture in Metal-Organic Frameworks

## Photoreactive Molecule Simlation
- 신약 개발시 빛을 쬐면 단백질에 고정되는 분자를 약물에 달아 약물이 붙는 단백질을 찾음
- Moderna - IBM

## Large-Scale Protein Simlation
- Cleveland Clinic x RIKEN x IBM
- 내가 만든 단백질이 약물에 얼마나 잘 붙는지 정밀 계산, 신약 후보 빠르게 고르기 가능

## Nuclear Fusion Material Simulation
- ORNL - Cleveland Clinic - IBM
- 핵융합소재를 개발하는 연구 - 삼중수소가 자연에 거의 없음. 리튬 + FLiBe 용융ㅇ염에 중성자를 반응시켜 생산

## OLED Emitter Simulation
- 삼성전자 - OTI Lumionics

## Early QPE on Real Hardware
- Quemix x Mitsui Kinzoku

## Road to QPE
- 궁극적인 형태의 목표
- QPE 를 실제로 돌리려면 무엇이 얼마나 필요하고 언제쯤 가능한가?

## Predicting Patient decline
- QuEra x University of western Australia

## Household Energy Demand Forecasting
- E.ON x WISER

## Soft Robot Motion Prediction
- Mitsubishi Chemical x Keio University

## KRAS InHibitor Discovery
- Insilico Medicine x Zapata AI x University of Toronto

## Ground-State Circuit Generation
- Pfizer x Quantinuum x NVIDAIA

## 과열된 기대보다, 지금 시작하는 준비가 가치를 만든다.
- NISQ는 오류 정정이 실증 단계에 들어선 FTQC로 가는 Bridge
- 지금은 주어진 한계를 보완하고 얕은 회로로 활용하면서, 가능성을 검증을 통해 FTQC 시기를 위한 역량을 쌓는 시기

- Hybrid Computing은 고전이 막히는 병목만 양자에 맡기는 NISQ 시대의 현실적인 전략
- QPU는 고전의 대체재가 아니라 CPU, GPU와 묶여 함께 활용되는 컴퓨팅 자원 - 어디에 무엇을 맡기느냐가 핵심

- Optimization, Simulation, Quantum + AI 로 Use Case가 넓어지는 중이나, 대부분은 아직 POT ~ POC 단계
- 산업보다 문제 구조가 적용 영역을 결정하며, 아직 양자 우위의 교차점 이전 단계

- 발표된 '성공'을 그대로 믿기보다 무엇과, 어떤 조건에서 비교했는지 따져 보는 것이 중요
- 같은 조건의 고전 비교, 필요 자원, 공개 검증 여부를 확인해야 과장된 기대와 실제 가치를 구분할 수 있음.

-> 양자 우위를 기다리기보다, 우리의 문제를 정의하고 검증하는 준비를 지금 시작할 때