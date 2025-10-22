## 캔버스와 렌더링 엔진의 관계

- Babylon.Engine은 HTML `<canvas>` 엘리먼트를 GPU와 연결하는 역할

```html
<canvas id="renderCanvas"></canvas>
<script>
const canvas = document.getElementById("renderCanvas");
const engine = new BABYLON.Engine(canvas, true);
</script>
```

- Babylon은 한 화면에 여러 개의 canvas도 관리 가능
(→ 미니맵, UI 등 만들 때 응용)

## 카메라 (Camera)

- 카메라는 3D 공간을 어떻게 볼지 정의

| 카메라 이름              | 특징                 | 예시          |
| ------------------- | ------------------ | ----------- |
| **ArcRotateCamera** | 중심을 축으로 도는 회전형 카메라 | 3D 모델 뷰어    |
| **UniversalCamera** | 1인칭 자유 이동          | FPS 스타일     |
| **FollowCamera**    | 특정 대상 추적           | 자동차, 캐릭터 게임 |
| **FreeCamera**      | 완전 수동 제어           | 사용자 정의      |

```js
const camera = new BABYLON.ArcRotateCamera(
  "arcCam",
  Math.PI / 2,
  Math.PI / 4,
  6,
  BABYLON.Vector3.Zero(),
  scene
);
camera.attachControl(canvas, true);

```

## 조명 (Light)

- 조명은 단순히 밝기뿐 아니라 그림자, 반사, 질감 표현에 영향

| 라이트 종류               | 설명                       |
| -------------------- | ------------------------ |
| **HemisphericLight** | 위쪽/아래쪽에서 오는 전반적인 빛 (기본용) |
| **DirectionalLight** | 태양광처럼 한 방향으로 오는 빛        |
| **PointLight**       | 한 점에서 모든 방향으로 빛 (전구처럼)   |
| **SpotLight**        | 원뿔 모양으로 쏘는 조명 (손전등)      |

```js
const light = new BABYLON.HemisphericLight(
  "light",
  new BABYLON.Vector3(0, 1, 0),
  scene
);
light.intensity = 0.9;

```

## 메쉬(Mesh) 만들기 — 3D 객체

- Babylon은 기본 도형을 쉽게 만들 수 있게 MeshBuilder API를 제공

```js
const sphere = BABYLON.MeshBuilder.CreateSphere("ball", {diameter: 2}, scene);
const box = BABYLON.MeshBuilder.CreateBox("box", {size: 1}, scene);
box.position.x = 3;
```

```js
box.rotation.y = Math.PI / 4;
box.scaling.set(2, 1, 1);

```

## 재질(Material)

- 메쉬는 단순한 “형태”이고, 색상이나 반사광은 Material이 담당

```js
const mat = new BABYLON.StandardMaterial("mat", scene);
mat.diffuseColor = new BABYLON.Color3(0, 0.5, 1);
mat.specularColor = new BABYLON.Color3(1, 1, 1);
sphere.material = mat;

```

## 텍스처 (Texture)

```js
const ground = BABYLON.MeshBuilder.CreateGround("ground", {width: 6, height: 6}, scene);
const groundMat = new BABYLON.StandardMaterial("gmat", scene);
groundMat.diffuseTexture = new BABYLON.Texture("textures/floor.jpg", scene);
ground.material = groundMat;
```

## 애니메이션 (Animation)

- 시간에 따라 속성을 바꾸는 방식

```js
const animation = new BABYLON.Animation(
  "moveBox",
  "position.x",
  30,  // FPS
  BABYLON.Animation.ANIMATIONTYPE_FLOAT,
  BABYLON.Animation.ANIMATIONLOOPMODE_CYCLE
);

const keys = [
  { frame: 0, value: 0 },
  { frame: 30, value: 2 },
  { frame: 60, value: 0 }
];
animation.setKeys(keys);
box.animations.push(animation);
scene.beginAnimation(box, 0, 60, true);
```

## 물리엔진 적용 (기초)

- 물리엔진을 활성화하면 중력, 충돌 등이 동작

```js
scene.enablePhysics(
  new BABYLON.Vector3(0, -9.81, 0),
  new BABYLON.CannonJSPlugin()
);

box.physicsImpostor = new BABYLON.PhysicsImpostor(
  box,
  BABYLON.PhysicsImpostor.BoxImpostor,
  { mass: 1, restitution: 0.9 },
  scene
);
```
- Cannon.js 또는 Ammo.js를 CDN으로 추가해야 작동

## GUI (2D UI를 3D에 띄우기)

- Babylon.js는 2D UI를 3D 위에 올릴 수 있는 GUI 시스템도 내장

```js
const gui = BABYLON.GUI.AdvancedDynamicTexture.CreateFullscreenUI("UI");
const button = BABYLON.GUI.Button.CreateSimpleButton("btn", "Click me!");
button.width = "150px";
button.height = "40px";
button.color = "white";
button.background = "blue";
button.onPointerUpObservable.add(() => alert("Clicked!"));
gui.addControl(button);
```

## 렌더 루프 제어

- engine.runRenderLoop()로 매 프레임마다 장면을 렌더링하지만,
필요하면 루프 안에 사용자 로직을 넣을 수 있음
```js
engine.runRenderLoop(() => {
  box.rotation.y += 0.01;
  scene.render();
});
```