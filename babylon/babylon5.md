## 렌더링 루프(Render Loop)의 개념

- Babylon.js는 매 프레임마다 장면(Scene)을 다시 그리는 반복 구조로 동작
- 이 과정을 60fps(초당 60번) 이상 반복함
→ 이렇게 돌아가는 걸 게임 루프(Game Loop) 라고 힘

## Scene Graph (장면 그래프)
- Babylon.js는 장면을 트리(Tree) 구조로 표현
- 모든 오브젝트(카메라, 빛, 메쉬)는 Scene의 하위 노드(Node)
- 부모-자식 관계가 있어, 부모가 회전하거나 이동하면 자식도 함께 움직임

```
Scene
 ├─ Light
 ├─ Camera
 └─ Mesh (Parent)
     └─ Mesh (Child)
```

## Transform (위치, 회전, 스케일)
- 모든 3D 객체는 3가지 변환 속성을 가집니다.

- position: 공간에서의 위치 (x, y, z)

- rotation 또는 rotationQuaternion: 회전

- scaling: 크기 조절

- 이 3가지를 합쳐서 월드 변환 행렬(World Matrix) 이 만들어지고, 이 행렬이 GPU로 전달되어 실제 좌표에 맞게 렌더링됨