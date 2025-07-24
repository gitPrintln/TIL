## ObjectInputStream & ObjectOutputStream

- 인스턴스를 입력하는 스트림 : ObjectInputStream - 인스턴스 직렬화
- 인스턴스를 출력하는 스트림 : ObjectOutputStream - 인스턴스 역직렬화

- 필터스트림과 비슷

```java
// ObjectOutputStream
~ implements java.io.Serializable {
    // 인스턴스를 직렬화를 위한 기본 조건인 Serializable 인터페이스의 구현
}
```

```java
// ObjectInputStream
ObjectInputStream // 객체 생성 후
readObject() // 호출하면됨 - 인스턴스 복원
```

- serializable을 구현하고 있지 않으면 인스턴스를 파일에 저장하지 않겠다고 생각하면 됨

#### transient를 선언하면 이 참조변수가 참조하는 대상은 저장하지 않겠다는 선언

```java
transient String s;
```