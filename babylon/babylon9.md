## PBR 추가 이론

#### 에너지 보존(Energy Conservation)

- 빛의 총량이 줄거나 늘지 않게 계산

#### Fresnel 효과 (프레넬 반사)

- 물체의 표면을 비스듬히 볼수록 반사가 강해짐

#### Microfacet 이론
- 표면은 완전히 매끈하지 않고 작은 면(미세면) 들로 구성되어 있다고 가정
- 이 미세면들이 빛을 각기 다른 방향으로 반사하면서 거칠기(roughness)가 표현

#### BRDF (Bidirectional Reflectance Distribution Function)

- 빛이 한 방향으로 들어와서 다른 방향으로 나갈 확률을 수학적으로 나타낸 함수.
- PBR은 이 BRDF를 기반으로 반사를 계산