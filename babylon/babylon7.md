## PBR 렌더링(Physically Based Rendering)

#### PBR : 빛이 실제 물리 법칙에 따라 반사되는 방식을 시뮬레이션하는 렌더링 기법

- 즉, “예쁘게 보이게”가 아니라 “현실처럼 보이게” 하는 접근.
- 빛의 입사 → 반사 → 흡수 → 산란 과정을 수학적으로 모델링

- Babylon.js의 PBRMaterial이나 GLTF 모델이 이 방식을 사용

#### 핵심 개념 4가지
| 개념                      | 뜻            | 역할                            |
| ----------------------- | ------------ | ----------------------------- |
| **Albedo (Base Color)** | 표면의 기본 색상    | 빛을 받기 전 순수한 물체의 색             |
| **Metalness**           | 금속성 (0~1)    | 빛 반사가 강하고, 색이 금속에 따라 변함       |
| **Roughness**           | 표면 거칠기 (0~1) | 낮을수록 반사가 선명(매끈), 높을수록 퍼짐(거칠음) |
| **Normal Map**          | 미세 요철        | 빛의 반사 각도를 세밀하게 바꿔 현실감 향상      |


#### 빛 반사 모델
- 빛이 어떻게 표면에서 반사되느냐를 두 가지로 나눠 생각
| 종류                            | 설명             | 예시          |
| ----------------------------- | -------------- | ----------- |
| **Diffuse Reflection (난반사)**  | 빛이 여러 방향으로 흩어짐 | 종이, 플라스틱, 벽 |
| **Specular Reflection (정반사)** | 빛이 한 방향으로 반사   | 금속, 거울, 유리  |

#### IBL (Image-Based Lighting)
- BR에서는 주변 환경(하늘, 건물 등)의 반사광도 계산
- 이를 위해 **HDR 환경맵 (Environment Map)**을 사용


```js
scene.environmentTexture = new BABYLON.CubeTexture("textures/env/environment.env", scene);
```

#### Babylon.js에서의 실제 사용 예
```js
const pbr = new BABYLON.PBRMaterial("pbr", scene);
pbr.albedoColor = new BABYLON.Color3(1, 0.8, 0.2); // 금빛
pbr.metallic = 1.0;  // 금속성
pbr.roughness = 0.2; // 매끄러움
sphere.material = pbr;
```
