## Protobuf(Protocol Buffers)

1. 개념: Google에서 개발한 데이터 직렬화(serialization) 형식 및 도구로, 주로 서비스 간 데이터 교환, 저장, 네트워크 전송 등에 사용

2. 사용 이유: 시스템 간 데이터를 교환할 때, JSON이나 XML 같은 텍스트 기반 형식은 가독성은 좋지만, 데이터가 크고 파싱 속도가 느리거나 타입 안정성이 떨어지는 특징이 있음 이에 반해 `protobuf`는
- 바이너리 형식 → 데이터 크기가 작음
- 빠른 파싱 → 네트워크 비용과 CPU 사용량을 절약
- 스키마 기반 → 컴파일 시 타입 안전성 확보

<br/>
이러한 특징으로 대용량 데이터 전송, 마이크로서비스, 모바일 앱 등에서 특히 많이 사용됨

<br/>

3. 스키마(정의 파일)
<br/>
.proto 확장자를 가진 텍스트 파일에 메시지 구조를 정의
<br/>

```proto
syntax = "proto3"
message Person {
    String name = 1;
    int32 id = 2;
    String email = 3;
}
```
<br/>

4. 컴파일
<br/>
`protoc`라는 컴파일러를 통해, `.proto` 파일을 원하는 언어 코드로 변환
<br/>
```bash
proto --java_out=. person.proto
```
- 이를 통해 Java, Python, C++, Go 등 다양한 언어로 생성 가능

<br/>

5. 데이터 직렬화/역직렬화
<br/>
- 직렬화(Serialization): 객체 -> 바이너리 데이터
- 역직렬화(Deserialization): 바이너리 데이터 -> 객체

```java
// Java 예제
Person john = Person.newBuilder()
    .setName("John Doe")
    .setId(1234)
    .setEmail("john@example.com")
    .build();

// 직렬화
byte[] data = john.toByteArray();

// 역직렬화
Person parsed = Person.parseFrom(data);
```

<br/>

6. 도구/라이브러리
<br/>
- protoc(공식 컴파일러) <br/>
- gRPC(원격 프로시저 호출) <br/>
- 각 언어별 protobuf 라이브러리 <br/>
<br/>

7. Json vs Protobuf

| 구분            | **Protobuf**                                 | **JSON**              |
| ------------- | -------------------------------------------- | --------------------- |
| **형식**        | **이진(binary) 형식**                            | **텍스트 기반 (문자열)**      |
| **크기**        | 매우 작음 (압축된 형태)                               | 큼 (문자열 포함, 중괄호/따옴표 등) |
| **속도**        | 빠름 (파싱, 직렬화/역직렬화 모두 빠름)                      | 느림 (문자열 파싱 오버헤드)      |
| **가독성**       | 사람이 읽기 어려움                                   | 사람이 읽고 수정하기 쉬움        |
| **스키마(정의)**   | `.proto` 파일로 명시적 정의 필요                       | 동적, 구조 자유로움           |
| **데이터 타입**    | 엄격한 타입 정의 (int32, string, repeated 등)        | 느슨한 타입 (런타임 확인)       |
| **진화(버전 호환)** | 필드 번호 기반이라 호환성 관리 용이                         | 구조 변경 시 파싱 에러 가능성 높음  |
| **용도**        | 시스템 간 고속 통신 (gRPC, 내부 API, IoT 등)            | Web API, 설정파일, 로그 등   |
| **지원 언어**     | 거의 모든 주요 언어 지원 (Java, Python, Go, C++, JS 등) | 거의 모든 언어 (표준 내장)      |
| **직렬화 결과**    | compact binary (바이너리 스트림)                    | readable JSON text    |

<br/>
8. gRPC와의 관계
gRPC는 protobuf를 기본 IDL(Interface Definition Language) 로 사용.
.proto 파일에 서비스(Service) 와 메시지(Message) 정의.

예:
```java
syntax = "proto3";

service Greeter {
  rpc SayHello (HelloRequest) returns (HelloReply);
}

message HelloRequest {
  string name = 1;
}

message HelloReply {
  string message = 1;
}
```
protoc로 컴파일하면 자동으로 클라이언트/서버 stub 코드가 생성됨.
HTTP/2 기반으로 스트리밍, 양방향 통신 가능.

<br/>

9. 버전 관리 & 호환성 설계 포인트
Protobuf는 스키마 진화(Schema Evolution) 가 강력한 장점

| 전략                     | 설명                            |
| ---------------------- | ----------------------------- |
| `optional`, `repeated` | 필드 존재 여부나 리스트 표현              |
| **필드 번호(tag number)**  | 변경하면 안 됨. 이름은 바꿔도 됨.          |
| **`reserved` 키워드**     | 삭제된 필드 번호나 이름을 예약 처리 → 재사용 방지 |
| **default 값 주의**       | 기본값(0, false, "")으로 초기화됨      |

<br/>
10. 실무에서 자주 부딪히는 이슈

| 주제                     | 요약                                               |
| ---------------------- | ------------------------------------------------ |
| **JSON ↔ protobuf 변환** | `JsonFormat` (Java), `MessageToJson()` 등으로 변환 가능 |
| **버전 불일치 문제**          | 서버/클라이언트의 protobuf 버전 다르면 직렬화 에러                 |
| **중첩 메시지 관리**          | 큰 시스템에선 메시지 구조를 분리(import)                       |
| **proto 패키지 충돌**       | 패키지 네임 관리 주의 (namespace 충돌 방지)                   |
| **optional 필드 처리**     | proto3는 optional을 나중에 도입했기 때문에 proto2와 다름        |
