## WebGL 기반 엔진이라는 것의 의미

Babylon.js는 사실상 **WebGL의 고수준 래퍼(wrapper)**이다.
WebGL은 OpenGL ES 2.0의 브라우저 버전으로,
GPU에 직접 “정점(Vertex)”과 “픽셀(Fragment)”을 명령하는 아주 저수준 API

Babylon.js는 이 복잡한 WebGL 코드를 직접 짜지 않아도 되도록 만들어진 **“엔진 레벨의 추상화 계층”**이다.
즉, Babylon.js는 **렌더링 파이프라인(Rendering Pipeline)**을 대신 관리해준다.

## Babylon.js의 내부 구조 이해

엔진의 핵심은 **렌더링 루프(Render Loop)**와 Scene Graph 구조이다.

- Scene Graph (장면 그래프) : 모든 객체(카메라, 메쉬, 라이트)는 트리 구조로 Scene에 연결

```java
Scene
 ├─ Camera
 ├─ Light
 └─ Mesh (Cube)
     └─ Child Mesh (e.g. attached weapon)

```

- 이런 식으로 상하 관계를 가질 수 있어서
“부모가 회전하면 자식도 함께 회전” 같은 트랜스폼(Transform)이 자동으로 적용

## 렌더링 파이프라인의 핵심

1. 입력 처리 (키보드, 마우스, 터치 등)

2. 물리 엔진 업데이트 (중력, 충돌 등)

3. 카메라 변환 계산

4. Scene 내 Mesh, Light, Material, Shadow 계산

5. Shader 코드 실행 → GPU 렌더링

이 모든 과정을 한 프레임(1/60초)마다 반복

## Babylon의 강점 (이론적 측면에서)
| 특징                               | 설명                                      |
| -------------------------------- | --------------------------------------- |
| **고수준 API**                      | 복잡한 WebGL 코드를 숨기고, 간단한 객체조작으로 가능        |
| **PBR(Material)**                | 실제 물리 기반 렌더링(빛 반사, 질감 표현이 사실적)          |
| **Physics Engine 내장**            | Cannon.js, Ammo.js, Havok Physics 통합 가능 |
| **WebXR 지원**                     | 브라우저에서 바로 VR/AR 컨텐츠 실행 가능               |
| **ShaderBuilder & NodeMaterial** | 개발자가 직접 셰이더 그래프를 구성 가능                  |
| **GPU Instancing**               | 수백~수천 개 객체를 효율적으로 렌더링                   |

## 렌더링 품질 관련 이론
- Babylon.js는 “PBR (Physically Based Rendering)”을 지원한다.
- 이건 현실의 빛 반사를 물리 법칙에 맞춰 시뮬레이션하는 방식이다.

- PBR을 이해하려면 아래 개념도 조금 알아두면 좋다:

Albedo: 기본 색상 (빛 반사 전 순수한 표면색)

Metalness / Roughness: 금속성, 거칠기

Normal Map: 표면 요철 표현

Environment Map: 주변 환경의 반사 시뮬레이션

## Babylon의 좌표계 (Coordinate System)
- Babylon은 기본적으로 **Y-Up, Right-Handed(오른손좌표계)**를 사용

X → 오른쪽

Y → 위쪽

Z → 앞쪽

Three.js는 기본적으로 Z-Up(왼손좌표계)

## Babylon.js가 WebGPU를 지원하는 이유

- 기존 WebGL은 오래된 OpenGL ES 2.0 기반이라, 병렬처리 성능이 제한적
- Babylon.js는 WebGPU 백엔드를 지원하여
새로운 그래픽 API (Vulkan/DX12급 GPU 접근)를 사용할 수 있게 해줌
- 더 빠른 연산, 더 사실적인 효과, 더 많은 객체를 처리 가능