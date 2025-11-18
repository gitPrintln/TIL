## 접근통제 모델(Access Control Models)

#### 1) DAC (임의적 접근통제, Discretionary Access Control)

- 소유자가 권한을 마음대로 줄 수 있는 방식
```js
특징

파일/자원의 소유자(owner) 가 권한을 부여

윈도우, 리눅스 파일 권한이 대표적

유연하지만 보안성이 낮다

예시

“이 파일 내가 만든거 → 너 읽기 권한 줄게”
```

#### 2) MAC (강제적 접근통제, Mandatory Access Control)

- 시스템이 강제적으로 접근 권한을 관리
```js
특징

등급(Level), 기밀성(Classification) 기반

사용자가 임의로 권한 수정 불가

보안성 가장 높음

군사·정부 시스템에서 사용

예시

기밀/일반/비밀/1급비밀 같은 등급이 있고

“기밀 등급 이상만 열람 가능”처럼 시스템이 통제
```

#### 3) RBAC (역할 기반 접근통제, Role-Based Access Control)

- 개별 사용자 → 역할(Role) → 권한 구조
```js
특징

기업/서비스에서 가장 많이 쓰임

사용자 권한 관리가 편함

역할이 곧 권한 묶음

예시

admin 역할 : read/write/delete

user 역할 : read only

직원이 부서 이동 → 역할만 바꾸면 권한 자동 변경
```

