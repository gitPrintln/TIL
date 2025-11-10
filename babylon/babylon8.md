## PBR을 다뤄보는 실제 관점

#### 기본 PBR 머티리얼 적용 예시
```js
const pbr = new BABYLON.PBRMaterial("metal", scene);
pbr.albedoColor = new BABYLON.Color3(0.9, 0.9, 0.9);
pbr.metallic = 1.0;
pbr.roughness = 0.2;
sphere.material = pbr;
```
metallic 올리면 → 금속 반짝임

roughness 올리면 → 거칠고 흐릿한 반사

albedoColor로 색상 조정