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
- protoc(공식 컴파일러)
- gRPC(원격 프로시저 호출)
- 각 언어별 protobuf 라이브러리