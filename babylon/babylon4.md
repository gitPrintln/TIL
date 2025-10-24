## 카메라 & 조명

#### ArcRotateCamera: 3D 공간 회전용

```js
const camera = new BABYLON.ArcRotateCamera(
  "arcCam",
  Math.PI / 2,   // alpha: 회전 각도 (좌우)
  Math.PI / 4,   // beta: 위아래 각도
  8,              // radius: 거리
  BABYLON.Vector3.Zero(), // 중심점(target)
  scene
);
camera.attachControl(canvas, true);
```
- 마우스로 드래그/줌 가능
- `camera.setTarget(mesh.position)` 하면 특정 객체 중심으로 초점 고정 가능

#### FollowCamera (특정 물체 따라가기)

```js
const followCam = new BABYLON.FollowCamera("follow", new BABYLON.Vector3(0, 3, -10), scene);
followCam.lockedTarget = box; // box를 따라감
followCam.radius = 10;        // 떨어진 거리
followCam.heightOffset = 2;   // 높이
scene.activeCamera = followCam;
```

## 조명(Light)

#### Directional Light (태양광)

```js
const sun = new BABYLON.DirectionalLight(
  "sun",
  new BABYLON.Vector3(-1, -2, -1),
  scene
);
sun.intensity = 1.2;
```

- 이 조명을 써야 그림자 생성 가능.

#### 그림자(Shadow) 만들기

```js
const shadowGen = new BABYLON.ShadowGenerator(1024, sun);
shadowGen.addShadowCaster(box); // 그림자 생길 대상 등록
ground.receiveShadows = true;   // 그림자를 받을 객체
```

#### 그림자 부드럽게 하기

```js
shadowGen.useBlurExponentialShadowMap = true;
shadowGen.blurKernel = 32;
```

## 반사(Reflection)

#### 거울 효과 (Mirror Texture)

```js
const mirrorMat = new BABYLON.StandardMaterial("mirrorMat", scene);
const mirrorTexture = new BABYLON.MirrorTexture("mirror", 512, scene, true);
mirrorTexture.mirrorPlane = new BABYLON.Plane(0, -1, 0, -2); // 평면 정의
mirrorMat.reflectionTexture = mirrorTexture;

const ground = BABYLON.MeshBuilder.CreateGround("ground", {width:6, height:6}, scene);
ground.material = mirrorMat;
mirrorTexture.renderList.push(box);
```

## 환경빛 & 하늘 (Ambient Light / Skybox)

#### Ambient Light (전반적인 환경조명)

```js
scene.ambientColor = new BABYLON.Color3(0.2, 0.2, 0.2);
```

- 약간 어두운 환경일 때 “전체적으로 밝기 보정”용.

#### Skybox (하늘/환경 반사 맵)

```js
const skybox = BABYLON.MeshBuilder.CreateBox("skyBox", {size: 1000}, scene);
const skyboxMaterial = new BABYLON.StandardMaterial("skyBoxMat", scene);
skyboxMaterial.backFaceCulling = false;
skyboxMaterial.reflectionTexture = new BABYLON.CubeTexture("textures/skybox/skybox", scene);
skyboxMaterial.reflectionTexture.coordinatesMode = BABYLON.Texture.SKYBOX_MODE;
skyboxMaterial.disableLighting = true;
skybox.material = skyboxMaterial;
```

