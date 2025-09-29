## 외부 API WebClient 의 경우

#### @MockBean으로 감싸서 given/when/then으로 테스트
#### JSON 직렬화/역직렬화는 ObjectMapper로 단위테스트

### JUnit5 (org.junit.jupiter.api.Assertions)에서 자주 쓰는 것들

- assertEquals(expected, actual)	값이 같은지 검증
- assertNotEquals(unexpected, actual)	값이 다른지 검증
- assertTrue(condition)	조건이 true인지
- assertFalse(condition)	조건이 false인지
- assertNull(object)	null인지
- assertNotNull(object)	null이 아닌지
- assertThrows(Exception.class, () -> …)	해당 코드가 예외를 던지는지
- assertDoesNotThrow(() -> …)	해당 코드가 예외를 던지지 않는지
- assertAll(…)	여러 검증을 한 번에 실행 (실패해도 전부 실행)
- assertIterableEquals(expectedIterable, actualIterable)	리스트나 Set 비교
- assertLinesMatch(expectedLines, actualLines)	문자열 리스트가 정규식 패턴 포함하여 일치하는지
- assertArrayEquals(expectedArray, actualArray)	배열 비교

```java
@Test
void testDto() {
    MyDto dto = new MyDto("ok", 123);

    assertAll(
        () -> assertEquals("ok", dto.getStatus()),
        () -> assertNotNull(dto.getValue()),
        () -> assertTrue(dto.getValue() > 100)
    );
}
```